# runpod-s2s-relay

Always-on CPU relay for a realtime speech-to-speech interpreter. This is the
only component the browser talks to. It holds the client WebSocket open,
decides where each spoken turn begins and ends, calls a scale-to-zero RunPod
serverless GPU endpoint to do the actual STT to LLM to TTS work, and streams
the resulting audio back to the browser as it is produced.

Companion repository: https://github.com/higeshogun/runpod-s2s-gpu-worker

## Why it is split this way

A GPU sitting idle waiting for somebody to speak is the most expensive thing
this system could possibly do. Splitting it in two means the only always-on
cost is a small CPU pod at roughly 0.08 USD/hr. The GPU endpoint scales to
zero between utterances and is billed per second of real work.

The split also keeps all the realtime behaviour - voice activity detection,
turn segmentation, resampling, and the client event protocol - in one
long-lived process, instead of trying to reimplement it inside a serverless
handler that only ever sees a single request at a time.

## End to end

1. The browser opens `wss://<POD_ID>-8765.proxy.runpod.net` and immediately
   sends a `session.update` carrying the interpreter persona in
   `instructions`.
2. The relay reads the language pair out of that instruction text and fires a
   warmup job at the GPU endpoint, so the cold start overlaps with the user
   getting settled rather than with their first sentence.
3. The browser streams microphone audio as 16 kHz mono PCM16, base64 encoded,
   inside `input_audio_buffer.append` events.
4. The relay runs webrtcvad over 30 ms frames. Once it has seen enough speech
   followed by 600 ms of silence it closes the turn, wraps the buffered audio
   in a WAV container and submits it to the GPU endpoint.
5. The worker streams back typed chunks: the input transcript first, then one
   `text` chunk plus several `audio` chunks per translated sentence.
6. The relay resamples each audio chunk from 24 kHz down to 16 kHz and
   forwards it as `response.output_audio.delta`, then ends the turn with
   `response.done`.

## Client protocol

Accepted from the client:

| Event | Meaning |
| --- | --- |
| `session.update` | `session.instructions` sets the persona for the session. An optional `session.languages` array overrides the language pair parsed from the instructions. |
| `input_audio_buffer.append` | `audio` is base64 PCM16, mono, 16 kHz. |

Emitted to the client:

| Event | Meaning |
| --- | --- |
| `input_audio_buffer.speech_started` | VAD opened a turn |
| `input_audio_buffer.speech_stopped` | VAD closed a turn |
| `response.created` | GPU job submitted |
| `conversation.item.input_audio_transcription.completed` | what the user said |
| `response.output_audio_transcript.delta` | translated text, one sentence at a time |
| `response.output_audio.delta` | base64 PCM16, mono, 16 kHz |
| `response.output_audio_transcript.done` | the full translated text |
| `response.done` | turn finished |
| `error` | `error.message` |

Three rules here are load-bearing, and each of them was a real bug first:

- Audio is emitted **once**, only as `response.output_audio.delta`. The client
  handles `response.audio.delta` in the same branch, so sending both makes
  every reply play twice.
- `response.done` is emitted **always**, including for empty transcripts and
  for failures. It is the only event that drains the client's audio player and
  returns it to listening, so skipping it wedges the session mid-turn.
- Empty audio deltas are never used as keepalives. The client treats any audio
  delta as playback starting: it flips to "speaking" and gates the microphone.
  The WebSocket ping keeps the connection alive instead.

## Turn segmentation

The client streams continuously and never sends a commit or a response
trigger, so every turn boundary is decided here.

| Constant | Value | Purpose |
| --- | --- | --- |
| `SILENCE_MS_TO_END_TURN` | 600 | silence needed to close a turn |
| `MIN_SPEECH_MS` | 400 | below this it is a blip, not an utterance |
| `MIN_SPEECH_RATIO` | 0.22 | buffers that are mostly room noise are dropped |
| `MAX_TURN_MS` | 15000 | force a flush so long speech still gets answered |
| `LEAD_PAD_MS` | 240 | audio kept before the first speech frame |
| `TAIL_PAD_MS` | 210 | audio kept after the last speech frame |
| `RMS_SPEECH_FLOOR` | 300 | webrtcvad alone triggers on breath and keyboard noise |

VAD aggressiveness is set to 3, the strictest level. Level 2 let breath, echo
and keyboard noise open turns, which sent whole GPU jobs for audio that
Whisper then transcribed as an empty string.

Barge-in is deliberately disabled. While a turn is in flight, incoming audio
is discarded rather than buffered, because buffering it produced phantom extra
turns the moment the GPU call returned.

## Resampling

Kokoro synthesizes at 24 kHz and the client plays at 16 kHz. `StreamResampler`
is a band-limited rational resampler: it upsamples by L, convolves with a
129-tap Blackman-windowed sinc low-pass, then decimates by M, carrying both
the filter delay line and the decimation phase across chunks so that sentence
boundaries stay click-free. `flush()` drains the delay line at the end of a
turn so the last few milliseconds of the final word actually reach the client.

The previous implementation interpolated linearly, which is not a low-pass
filter at all: everything above 8 kHz folded back down into the audible band,
and that aliasing is what gave the voice its harsh, metallic edge.

## Language handling

A session only ever involves two languages, but Whisper's language ID runs
over all 99 and gets short clips wrong often enough to matter - a Japanese
sentence detected as Korean comes back as garbage. The relay therefore parses
the pair out of the client's instruction text, which contains one sentence per
direction of the form "When someone speaks X, say only the Y translation.",
and passes the resulting codes to the worker so detection can be restricted to
those two.

If the instructions cannot be parsed, `SESSION_LANGUAGES` is used instead.

## Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `RUNPOD_ENDPOINT_ID` | required | the serverless endpoint id |
| `RUNPOD_API_KEY` | required | never commit this; keep it out of templates too |
| `RELAY_HOST` | `0.0.0.0` | |
| `RELAY_PORT` | `8765` | |
| `SESSION_LANGUAGES` | `en,ja` | fallback pair when instructions cannot be parsed |
| `WARMUP_ON_CONNECT` | `1` | set to `0` to disable the warmup job |

## Deploying

The pod is a stock `runpod/base:0.7.0-ubuntu2004` CPU pod with HTTP port 8765
exposed. A RunPod pod template named `captionify-s2s-relay` captures that
configuration, including the non-secret environment variables and the
bootstrap instructions in its README.

Recommended instance: cpu3g, 2 vCPU, 8 GB RAM, 5 GB container disk. The relay
is I/O bound and spends nearly all of its time waiting on the GPU endpoint, so
a larger instance buys nothing.

First-time setup on a fresh pod:

```
git clone https://github.com/higeshogun/runpod-s2s-relay.git /root/runpod-s2s-relay
cd /root/runpod-s2s-relay
python3.13 -m pip install -r requirements.txt
export RUNPOD_ENDPOINT_ID=<endpoint id>
export RUNPOD_API_KEY=<api key>
tmux new -s relay
python3.13 relay.py
```

Use `python3.13` explicitly. The stock `python3` on this image is 3.8 and does
not have aiohttp, websockets, numpy or webrtcvad available, so plain
`python3 relay.py` fails with a ModuleNotFoundError.

Detach from tmux with ctrl-b then d.

## Updating

There is no automatic deploy. Pull and restart by hand:

```
tmux attach -t relay
# ctrl-c to stop the running relay
git pull && python3.13 relay.py
```

## Operational notes

- Startup logs `Relay listening on ws://0.0.0.0:8765`. Per-turn logs include
  the transcript, the detected language, time to first audio out, and whether
  the worker actually streamed.
- A cold GPU worker takes roughly 40 to 60 seconds to first audio; a warm one
  takes 1.5 to 4.5 seconds. The endpoint's idle timeout decides how often you
  pay the cold path.
- `GPU_CALL_TIMEOUT_S` is 300 s, deliberately long enough to survive a cold
  start that includes pulling a new image.
