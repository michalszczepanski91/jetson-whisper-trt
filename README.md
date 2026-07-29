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
├── src/                             # (unused placeholder — see "Talking to
│                                    # embedded-ai-chain" below for why this
│                                    # repo stays standalone, no shared code)
│
├── tests/
│   └── test_live_transcribe.py     # BridgeClient unit tests (mic/GPU parts
│                                    # need real hardware, not unit-testable)
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

## Talking to `embedded-ai-chain`

`embedded-ai-chain`'s vision producers (`yolo_producer.py`) run in-process on the host,
sharing an in-memory `scene_state.py` object — no IPC, since YOLO writes ~30 times a
second and any per-frame round-trip would reintroduce the latency `scene_state.py`
exists to avoid. STT doesn't have that constraint: it produces one discrete event per
spoken utterance (a few times a minute at most), so it doesn't need to run in the same
process, or even avoid IPC at all.

That matters because `torch` itself doesn't reliably run in `embedded-ai-chain`'s own
`.venv` on this device — a plain CUDA matmul fails with `CUBLAS_STATUS_ALLOC_FAILED`
regardless of which `nvidia-cublas-cu12` version is pinned, most likely because generic
PyPI CUDA packages (built for standard desktop/server GPUs) don't reliably work against
Jetson's own `nvgpu` driver stack — which is exactly why NVIDIA maintains a separate,
Jetson-tested pip index in the first place. This container doesn't have that problem
(NVIDIA's own image, built and validated for Jetson).

So: `live_transcribe.py` sends each transcript to `embedded-ai-chain`'s host process
(`src/stt_bridge.py`) over a plain TCP socket — `docker-compose.yml` uses
`network_mode: host` for both services, so `127.0.0.1` on the host is directly
reachable from inside the container, no port mapping needed. `BridgeClient`
(in `scripts/live_transcribe.py`) connects lazily and reconnects on failure rather
than raising — the host orchestrator may not be up yet, or may restart, and a live
mic session should keep running (and keep printing/logging locally) regardless.
Controlled by `configs/pipeline.yaml`'s `bridge:` section.

The received transcripts are **not** written into `scene_state` — that store is
vision-only (see its own docstring). They're delivered via a plain `queue.Queue` for
whatever consumes them (the orchestrator, not yet built).

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
(and optionally logs to `output/transcript.log`, see `configs/pipeline.yaml`), and —
if `bridge.enabled` in `configs/pipeline.yaml` — sends each transcript to
`embedded-ai-chain`'s host process. See "Talking to embedded-ai-chain" below.

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

### 8. Export to ONNX
```bash
make export-onnx MODEL=base.en
```
`whisper_trt` only ever builds ONNX in-memory as an intermediate step toward a
TensorRT engine — it never saves the file. This reuses its own encoder/decoder
wrapper modules and dummy-input shapes to export two portable ONNX files (encoder
and decoder are separate graphs — the decoder runs once per generated token,
taking the encoder's fixed output as an input) to
`/opt/models/jetson-whisper-trt/onnx/`, runnable with plain `onnxruntime`, no
TensorRT/torch2trt required.

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
