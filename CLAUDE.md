# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WhisperTRT speech-to-text pipeline for NVIDIA Jetson Orin. Wraps
[NVIDIA-AI-IOT/whisper_trt](https://github.com/NVIDIA-AI-IOT/whisper_trt) (pinned to a
specific commit, not tracking `main`) rather than reimplementing Whisper's
encoder-decoder TensorRT conversion from scratch. Sibling project to
`jetson-yolov8-trt`, same containerized/Makefile-driven shape.

## Common Commands

All workflows are container-based. Run these from the repo root:

```bash
# Build images
make build-dev          # Dev image: compiles torch2trt, installs whisper_trt, adds Jupyter/pytest/etc.
make build              # Production image
make shell               # Interactive dev shell

# Environment
make validate            # CUDA/TensorRT/torch2trt/whisper_trt/pyaudio sanity check
make list-devices        # Enumerate PortAudio input devices (needed to set configs/pipeline.yaml)

# Transcription
make transcribe-file AUDIO=path/to.wav                       # default: base.en, whisper_trt backend
make transcribe-file AUDIO=path/to.wav MODEL=tiny.en BACKEND=whisper
make benchmark AUDIO=path/to.wav MODEL=base.en                # whisper vs whisper_trt latency+memory
make live                                                      # mic -> VAD -> transcribe loop (dev container)
make run                                                        # same, production image

# Cleanup
make clean                # Remove __pycache__, .pyc
make clean-docker         # Remove Docker images
```

## Architecture

**Data flow (live_transcribe.py):**
```
Mic (PyAudio, single channel) → Silero VAD (ONNX Runtime, bundled in whisper_trt.vad)
  → speech segment buffering → WhisperTRT.transcribe() → stdout / output/transcript.log
```

**Why wrap `whisper_trt` instead of a bespoke PT→ONNX→TensorRT export** (the approach
used in `jetson-yolov8-trt`): YOLOv8 is a single-pass detector, straightforward to
hand-export. Whisper is encoder-decoder with autoregressive decoding — far more
error-prone to convert by hand. `whisper_trt` already solves this and bundles Silero
VAD; we depend on it as a pinned git package rather than forking it.

**Configuration:** `configs/pipeline.yaml` controls mic device selection (substring
match on PortAudio device name — see `make list-devices`), model size (`tiny.en` /
`base.en` / `small.en`), VAD speech threshold, and output mode.

**Container setup:**
- Production: `Dockerfile` at repo root
- Development: `docker/Dockerfile.dev` — adds Jupyter (port 8888), pytest, black, isort, flake8
- `docker-compose.yml` defines `pipeline` (production) and `dev` (volume-mounted for
  live editing) — both need `ipc: host` (NVIDIA base image's default 64MB `/dev/shm`
  silently deadlocks PyTorch multiprocessing otherwise) and `/dev/snd` + `group_add:
  audio` for microphone access.

**Shared storage:** the TensorRT engine cache (and Silero VAD ONNX download) that
`whisper_trt` normally keeps at `~/.cache/whisper_trt` is bind-mounted from
`/opt/models/jetson-whisper-trt` instead — shared across containers/users on this
machine rather than rebuilt per-clone. Same convention as `jetson-yolov8-trt`'s model
checkpoints; see the `embedded-ai-chain` parent repo's `docs/shared-storage.md`.

## Current State

Verified working end-to-end 2026-07-29: `make validate` (16/16 checks, including
detecting the Jabra mic inside the container), `make transcribe-file` (correct
transcription of a real speech sample — the standard Harvard Sentences test audio),
and `make benchmark` (whisper_trt 2.4x faster than the PyTorch baseline on `tiny.en`,
in line with upstream's own published numbers) all pass. See the README's
"Compatibility patches" note for the four fixes this took against this base image's
much-newer PyTorch than `whisper_trt`/`torch2trt` were built against.

`scripts/live_transcribe.py` (mic → VAD → transcribe loop) is implemented but not yet
smoke-tested against a live microphone in this session — it's a single-process
simplification of `whisper_trt`'s multiprocess ReSpeaker-array example, adapted for
this project's actual hardware (single-channel Jabra USB mic, confirmed in Phase 0 of
the `embedded-ai-chain` parent repo). Not yet wired into that repo's `scene_state.py`
as a producer — that integration is Phase 1 work in the parent repo, not this one.

## Key Dependencies

- **Base image:** `nvcr.io/nvidia/pytorch:26.03-py3` (JetPack 6.x, CUDA 12.6, TensorRT 10.3) —
  same base as `jetson-yolov8-trt`, confirmed working on this exact hardware/JetPack combo
- **ASR:** `openai-whisper==20240930` (pinned to match `whisper_trt`'s last commit —
  `whisper_trt` imports Whisper's internal model classes directly, not a stable public API)
- **TensorRT conversion:** `torch2trt` (pinned commit `4e820ae`), `whisper_trt` (pinned
  commit `268eff1`) — both MIT licensed, installed from source (not on PyPI)
- **VAD:** Silero VAD via ONNX Runtime, bundled in `whisper_trt.vad`
- **Audio:** `pyaudio` (needs `portaudio19-dev` at the OS level, in both Dockerfiles)

## Hardware Notes

- Designed for Jetson Orin (ARM64). TensorRT engines are device-specific — an engine
  built on one Jetson variant will not run on another (same caveat as `jetson-yolov8-trt`).
- Mic is a Jabra EVOLVE 30 II over USB — the Logitech C930e webcam's built-in mic is
  stereo-only at the hardware level and not usable for direct mono capture (see the
  parent repo's `docs/environment.md`).
