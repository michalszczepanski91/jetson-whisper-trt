# ─────────────────────────────────────────────────────────────────────────────
#  Dockerfile — Jetson WhisperTRT (Production)
#  Same base image as jetson-yolov8-trt — confirmed working on R36.4.7 / driver 540.4.0
# ─────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/pytorch:26.03-py3

LABEL maintainer="jetson-whisper-trt"
LABEL description="WhisperTRT STT pipeline for Jetson Orin, wrapping NVIDIA-AI-IOT/whisper_trt"
LABEL version="0.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV WORKSPACE=/workspace
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

RUN apt-get update && apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    git \
    cmake \
    curl \
    wget \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

# openai-whisper pinned to the release contemporaneous with whisper_trt's last
# commit (2024-10-15) — whisper_trt imports whisper's internal model classes
# directly (whisper.model.LayerNorm/Linear/Whisper etc.), which are not a
# stable public API, so a newer whisper release risks breaking the TensorRT
# conversion in ways that won't show up until you actually try to load a model.
#
# --no-build-isolation: openai-whisper's setup.py is old-style and imports
# pkg_resources at build time; pip's isolated build env picks up a setuptools
# that doesn't bundle pkg_resources by default, so it fails unless we build
# using the base image's own (already-working) setuptools instead.
RUN pip3 install --no-cache-dir --no-build-isolation openai-whisper==20240930

# Patch: force MultiHeadAttention.use_sdpa off by default. whisper_trt (last
# commit Oct 2024) builds its TensorRT decoder engine expecting an explicit
# `mask` input tensor — true for whisper's original manual softmax attention
# path, but this whisper version defaults to the newer scaled_dot_product_attention
# fast path instead, which applies causal masking internally via `is_causal`
# and never uses `mask` as an actual traced tensor input. That mismatch is
# exactly what produces "Invalid tensor name: mask" TensorRT errors at
# inference time and garbled, repetitive transcriptions (masking silently not
# applied). whisper.model already ships a `disable_sdpa()` context manager
# for this; we just flip its default instead of requiring every call site
# (including inside whisper_trt, which was never updated for this) to use it.
RUN sed -i 's/    use_sdpa = True/    use_sdpa = False/' \
    /usr/local/lib/python3.12/dist-packages/whisper/model.py

RUN pip3 install --no-cache-dir \
    onnxruntime==1.20.1 \
    pyaudio==0.2.14 \
    psutil==5.9.8 \
    PyYAML \
    rich==13.7.1 \
    jiwer==3.0.4

# onnx_graphsurgeon — NVIDIA's private package index, not on public PyPI. Needed
# by torch2trt at engine-build time (whisper_trt's decoder conversion path hits
# `import onnx_graphsurgeon` inside torch2trt.torch2trt()), not just at import time,
# so this only surfaces the first time you actually build an engine.
RUN pip3 install --no-cache-dir onnx onnx-graphsurgeon --extra-index-url https://pypi.ngc.nvidia.com

# torch2trt — not on PyPI, pinned to the last commit (2024-05-03). Skipping the
# optional --plugins build: whisper_trt's model.py only uses standard
# torch2trt converters (Linear, LayerNorm, matmul, softmax, etc.), no custom
# plugin ops.
#
# Patch: torch2trt's ONNX export path (torch2trt.py) calls torch.onnx.export()
# with the legacy `dynamic_axes` argument. This base image's PyTorch (2.11,
# March 2026) defaults torch.onnx.export() to the newer dynamo-based exporter,
# whose dynamic_axes->dynamic_shapes compatibility shim is broken for this
# call shape ("ValueError: treespec.unflatten(leaves)..."). Force the legacy
# TorchScript-based exporter, which still supports dynamic_axes correctly.
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt \
    && cd /opt/torch2trt \
    && git checkout 4e820ae31b4e35d59685935223b05b2e11d47b03 \
    && sed -i 's/opset_version=onnx_opset/opset_version=onnx_opset,\n            dynamo=False/' torch2trt/torch2trt.py \
    && python3 setup.py install

# whisper_trt itself (MIT licensed) — pinned to its last commit for reproducibility.
# --no-build-isolation again: its setup.py imports whisper_trt.__version__, which
# forces Python to run whisper_trt/__init__.py first (imports .model, which does
# `from whisper import load_model`) — `whisper` isn't visible inside a fresh
# isolated build sandbox, only in the outer env where we just installed it.
RUN git clone https://github.com/NVIDIA-AI-IOT/whisper_trt /opt/whisper_trt \
    && cd /opt/whisper_trt \
    && git checkout 268eff10a1e38118a2734745b9db14f7419a08a5 \
    && pip3 install --no-cache-dir --no-build-isolation .

WORKDIR ${WORKSPACE}

COPY src/       ${WORKSPACE}/src/
COPY scripts/   ${WORKSPACE}/scripts/
COPY configs/   ${WORKSPACE}/configs/

RUN chmod +x ${WORKSPACE}/scripts/*.py

CMD ["python3", "scripts/live_transcribe.py", "--config", "configs/pipeline.yaml"]
