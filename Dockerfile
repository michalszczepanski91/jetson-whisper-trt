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
RUN pip3 install --no-cache-dir \
    openai-whisper==20240930 \
    onnxruntime==1.20.1 \
    pyaudio==0.2.14 \
    psutil==5.9.8 \
    PyYAML \
    rich==13.7.1

# torch2trt — not on PyPI, pinned to the last commit (2024-05-03). Skipping the
# optional --plugins build: whisper_trt's model.py only uses standard
# torch2trt converters (Linear, LayerNorm, matmul, softmax, etc.), no custom
# plugin ops.
RUN git clone https://github.com/NVIDIA-AI-IOT/torch2trt /opt/torch2trt \
    && cd /opt/torch2trt \
    && git checkout 4e820ae31b4e35d59685935223b05b2e11d47b03 \
    && python3 setup.py install

# whisper_trt itself (MIT licensed) — pinned to its last commit for reproducibility.
RUN git clone https://github.com/NVIDIA-AI-IOT/whisper_trt /opt/whisper_trt \
    && cd /opt/whisper_trt \
    && git checkout 268eff10a1e38118a2734745b9db14f7419a08a5 \
    && pip3 install --no-cache-dir .

WORKDIR ${WORKSPACE}

COPY src/       ${WORKSPACE}/src/
COPY scripts/   ${WORKSPACE}/scripts/
COPY configs/   ${WORKSPACE}/configs/

RUN chmod +x ${WORKSPACE}/scripts/*.py

CMD ["python3", "scripts/live_transcribe.py", "--config", "configs/pipeline.yaml"]
