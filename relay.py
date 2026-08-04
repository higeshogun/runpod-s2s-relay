import asyncio
import base64
import io
import json
import os
import wave

import aiohttp
import webrtcvad
import websockets

RUNPOD_ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]
RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
RUNSYNC_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}/runsync"

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_BYTES = int(SAMPLE_RATE * FRAME_MS / 1000) * 2  # 16-bit mono PCM
SILENCE_MS_TO_END_TURN = 700
MIN_SPEECH_MS = 250
MAX_HISTORY_TURNS = 6

vad = webrtcvad.Vad(2)  # 0-3, higher = stricter about what counts as speech


def pcm_to_wav_bytes(pcm_bytes):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


async def call_gpu_endpoint(session, wav_bytes, history):
    payload = {"input": {"audio_base64": base64.b64encode(wav_bytes).decode(), "history": history}}
    headers = {"Authorization": f"Bearer {RUNPOD_API_KEY}", "Content-Type": "application/json"}
    async with session.post(RUNSYNC_URL, json=payload, headers=headers,
                             timeout=aiohttp.ClientTimeout(total=60)) as resp:
        data = await resp.json()
        return data.get("output", {})


class TurnBuffer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.frames = []
        self.triggered = False
        self.silence_ms = 0
        self.speech_ms = 0

    def add_frame(self, frame, is_speech):
        if is_speech:
            self.triggered = True
            self.frames.append(frame)
            self.speech_ms += FRAME_MS
            self.silence_ms = 0
        elif self.triggered:
            self.frames.append(frame)
            self.silence_ms += FRAME_MS

    def finished(self):
        return self.triggered and self.silence_ms >= SILENCE_MS_TO_END_TURN and self.speech_ms >= MIN_SPEECH_MS

    def audio_bytes(self):
        return b"".join(self.frames)


async def handle_client(websocket):
    history = []
    turn = TurnBuffer()
    leftover = b""

    async with aiohttp.ClientSession() as session:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue  # ignore text control frames for now

            leftover += message
            while len(leftover) >= FRAME_BYTES:
                frame, leftover = leftover[:FRAME_BYTES], leftover[FRAME_BYTES:]
                is_speech = vad.is_speech(frame, SAMPLE_RATE)
                turn.add_frame(frame, is_speech)

                if turn.finished():
                    wav_bytes = pcm_to_wav_bytes(turn.audio_bytes())
                    turn.reset()

                    output = await call_gpu_endpoint(session, wav_bytes, history)
                    transcript = output.get("transcript", "")
                    if not transcript:
                        continue

                    response_text = output.get("response_text", "")
                    history.extend([{"role": "user", "content": transcript},
                                     {"role": "assistant", "content": response_text}])
                    history[:] = history[-MAX_HISTORY_TURNS * 2:]

                    await websocket.send(json.dumps({"type": "transcript", "text": transcript}))
                    await websocket.send(json.dumps({"type": "response_text", "text": response_text}))

                    audio_b64 = output.get("response_audio_base64", "")
                    if audio_b64:
                        await websocket.send(base64.b64decode(audio_b64))


async def main():
    host = os.environ.get("RELAY_HOST", "0.0.0.0")
    port = int(os.environ.get("RELAY_PORT", "8765"))
    async with websockets.serve(handle_client, host, port, max_size=None):
        print(f"Relay listening on ws://{host}:{port}")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
