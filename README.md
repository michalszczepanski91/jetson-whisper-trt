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
