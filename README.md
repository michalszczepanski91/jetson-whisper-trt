# jetson-whisper-trt

WhisperTRT STT pipeline for NVIDIA Jetson Orin. Wraps
[NVIDIA-AI-IOT/whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) (MIT licensed) —
which itself converts OpenAI Whisper's encoder/decoder to TensorRT via
[torch2trt](https://github.com/NVIDIA-AI-IOT/torch2trt) — in the same containerized,
Makefile-driven shape as the sibling [jetson-yolov8-trt](https://github.com/michalszczepanski91/jetson-yolov8-trt)
project.

Not a fork of `whisper_trt` — it's pulled in as a pinned git dependency inside the
Docker image (see `Dockerfile`), the same way `jetson-yolov8-trt` depends on
`ultralytics` as a package rather than forking it.

---

## Project Structure

```
jetson-whisper-trt/
│
├── docker/
│   └── Dockerfile.dev              # Dev image (adds Jupyter, pytest, etc.)
│
├── src/                             # (placeholder) STT producer glue for the
│                                    # embedded-ai-chain scene-state integration
│
├── scripts/
│   ├── validate_install.py         # Env sanity check (CUDA, TensorRT, torch2trt,
│   │                                # whisper_trt, pyaudio device enumeration)
│   ├── transcribe_file.py          # Transcribe a single WAV file
│   ├── benchmark.py                # Compare whisper / whisper_trt latency+memory
│   └── live_transcribe.py          # Mic -> Silero VAD -> WhisperTRT loop
│
├── configs/
│   └── pipeline.yaml               # Mic device, model size, VAD threshold, output mode
│
├── Dockerfile                      # Production image
├── docker-compose.yml              # Container orchestration
└── Makefile                        # Dev shortcuts
```

---

## Why wrap `whisper_trt` instead of building our own PT→ONNX→TensorRT export

YOLOv8 is a single-pass detector — straightforward to export PT→ONNX→TensorRT by hand
(see `jetson-yolov8-trt`). Whisper is an encoder-decoder model with autoregressive
decoding; hand-rolling that conversion (KV-cache handling, separate encoder/decoder
engines, the decode loop itself) is materially harder and riskier than reusing
NVIDIA's own Jetson-specific implementation. `whisper_trt` already handles all of
that, plus bundles a Silero VAD (ONNX Runtime) implementation we'd otherwise have to
add separately.

`torch2trt` and `whisper_trt` are both pinned to specific commits in the `Dockerfile`
(not tracking `main`) for reproducibility — `whisper_trt` imports Whisper's internal
model classes directly (`whisper.model.LayerNorm`/`Linear`/`Whisper`, not a stable
public API), so `openai-whisper` is pinned to the release contemporaneous with
`whisper_trt`'s last commit to minimize the risk of internal-API drift.

Even with that pin, the base image's PyTorch (2.11, March 2026) is far newer than
anything `whisper_trt`/`torch2trt` (last touched mid/late-2024) were built or tested
against. Getting a real end-to-end transcription working required four patches,
applied in the `Dockerfile`s via `sed` against the installed packages (not forks —
see each patch's inline comment for the full explanation):

1. **`pip3 install --no-build-isolation`** for both `openai-whisper` and `whisper_trt` —
   their old-style `setup.py`s need packages (`pkg_resources`, transitively `whisper`
   itself) that a fresh pip isolated-build sandbox doesn't have.
2. **`onnx-graphsurgeon`** — a `torch2trt` runtime dependency not on public PyPI
   (NVIDIA's own index), only needed the first time an engine is actually built.
3. **`dynamo=False`** on `torch2trt`'s internal `torch.onnx.export()` call — this
   PyTorch defaults to the newer dynamo-based ONNX exporter, whose
   `dynamic_axes`→`dynamic_shapes` shim is broken for `torch2trt`'s call shape.
4. **`MultiHeadAttention.use_sdpa = False`** (whisper's own toggle, just flipping its
   default) — this whisper version's default fast-attention path applies causal
   masking internally via `is_causal` rather than an explicit `mask` tensor, but
   `whisper_trt`'s TensorRT engine-building code expects `mask` as an explicit input.
   Without this, engines build "successfully" but produce garbled, repetitive
   transcriptions and `Invalid tensor name: mask` errors at inference time — a much
   nastier failure mode than a build error, since it doesn't fail loudly.

If `whisper_trt` or `torch2trt` ever get updated upstream, these patches should be
revisited — they may no longer be needed, or may need adjusting for a new pinned commit.

---

## Workflow

### 1. Build the dev image
```bash
make build-dev
```
First build compiles `torch2trt` from source and installs `whisper_trt` — takes a
few minutes.

### 2. Check the environment
```bash
make validate
```

### 3. Find your microphone
```bash
make list-devices
```
`configs/pipeline.yaml`'s `audio.device_name_contains` does a substring match against
PortAudio device names (not a raw ALSA `hw:X,Y` index — those don't map 1:1 onto
PortAudio's own enumeration inside the container). Defaults to `"jabra"`, matching the
Jabra EVOLVE 30 II confirmed in `embedded-ai-chain`'s Phase 0 environment verification
as the only mic that captures native mono 16kHz without resampling.

### 4. Transcribe a file
```bash
make transcribe-file AUDIO=/path/to/speech.wav
make transcribe-file AUDIO=/path/to/speech.wav MODEL=tiny.en BACKEND=whisper
```
First call for a given model size builds and caches the TensorRT engine (can take a
while); cached at `/opt/models/jetson-whisper-trt` (shared storage, see
`embedded-ai-chain`'s `docs/shared-storage.md` — same convention as
`jetson-yolov8-trt`'s model checkpoints), so subsequent runs and other projects on
this machine reuse it.

### 5. Benchmark
```bash
make benchmark AUDIO=/path/to/speech.wav MODEL=base.en
```
Compares `whisper` (PyTorch baseline) vs `whisper_trt`. Add `faster_whisper` to the
Dockerfile's pip install list for a 3-way comparison (not installed by default to
keep the image lean).

### 6. Live transcription
```bash
make live          # inside the dev container, for iterating
make run           # production image
```
Speaks into the configured mic, VAD-gates on speech, transcribes each segment, prints
(and optionally logs to `output/transcript.log`, see `configs/pipeline.yaml`).

### 7. Accuracy (WER) on LibriSpeech test-clean
```bash
make validate-librispeech MODEL=base.en LIMIT=50   # quick sanity check
make validate-librispeech MODEL=base.en             # full 2620-utterance set (~20 min)
```
Dataset: `/opt/datasets/librispeech` (staged the same way as `jetson-yolov8-trt`'s
COCO val2017 — see `embedded-ai-chain`'s `docs/shared-storage.md`). Scores against
OpenAI's own `EnglishTextNormalizer` (matches how the Whisper paper computes WER —
without it, casing/punctuation differences inflate the error count without reflecting
real transcription mistakes). Results appended to `output/librispeech_results.json`.

LibriSpeech `test-clean` is clean, studio-mic read audiobook speech — it validates the
harness against a standard, citable number, but doesn't tell you how this actually
behaves on this robot's own mic, at conversational distance, with background noise,
or whether it hallucinates during silence. That's what the bespoke test corpus in
`embedded-ai-chain`'s `docs/TODO.md` §2b is for — a separate, more effortful task
(real recording, not a download).

---

## Other Commands

```bash
make shell                         # Interactive bash shell inside dev container
make validate                      # Verify CUDA, TensorRT, torch2trt, whisper_trt are installed
make clean                         # Remove __pycache__, .pyc (not /opt/models — shared storage)
make clean-docker                  # Remove Docker images
```

---

## Hardware

- **Device:** NVIDIA Jetson Orin (ARM64), L4T R36.4.7
- **JetPack:** 6.x (CUDA 12.6, TensorRT 10.3)
- **Mic:** Jabra EVOLVE 30 II (USB), confirmed native mono 16kHz capture
- **GPU access:** requires `--runtime nvidia` (set in `docker-compose.yml`)
- **`ipc: host`** is required on both compose services — the NVIDIA base image's
  default 64MB `/dev/shm` silently deadlocks PyTorch multiprocessing (see
  `jetson-yolov8-trt`'s `docker-compose.yml` for the incident that surfaced this)

---

## License note

`whisper_trt` and `torch2trt` are both MIT licensed (NVIDIA). This repo's own code
(everything under `scripts/`, `configs/`, `src/`, the `Dockerfile`s) follows suit —
no AGPL-style complication here, unlike the Ultralytics dependency in
`jetson-yolov8-trt`.
