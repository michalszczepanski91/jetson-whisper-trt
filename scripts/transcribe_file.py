#!/usr/bin/env python3
"""Transcribe a single audio file with a chosen backend.

Adapted from NVIDIA-AI-IOT/whisper_trt's examples/transcribe.py (MIT licensed)
https://github.com/NVIDIA-AI-IOT/whisper_trt — restructured to match this
repo's argparse/timing conventions (see jetson-yolov8-trt/scripts/infer_image.py).

Usage (inside dev container):
    python scripts/transcribe_file.py speech.wav
    python scripts/transcribe_file.py speech.wav --model tiny.en --backend whisper
"""

import argparse
import json
import time

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("audio", help="Path to a .wav file")
    p.add_argument("--model", choices=["tiny.en", "base.en", "small.en"], default="base.en")
    p.add_argument("--backend", choices=["whisper_trt", "whisper", "faster_whisper"], default="whisper_trt")
    return p.parse_args()


def load(model_name: str, backend: str):
    if backend == "whisper_trt":
        from whisper_trt import load_trt_model
        return load_trt_model(model_name), False
    elif backend == "whisper":
        from whisper import load_model
        return load_model(model_name), False
    elif backend == "faster_whisper":
        from faster_whisper import WhisperModel
        return WhisperModel(model_name), True
    raise ValueError(f"unknown backend: {backend}")


def main():
    args = parse_args()

    t0 = time.perf_counter()
    model, is_faster_whisper = load(args.model, args.backend)
    load_s = time.perf_counter() - t0  # includes first-run TensorRT engine build for whisper_trt

    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    result = model.transcribe(args.audio)
    if is_faster_whisper:
        segs, _info = result
        result = {"text": "".join(seg.text for seg in segs)}
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    transcribe_s = time.perf_counter() - t0

    print(json.dumps({
        "backend": args.backend,
        "model": args.model,
        "text": result["text"],
        "load_seconds": load_s,
        "transcribe_seconds": transcribe_s,
    }, indent=2))


if __name__ == "__main__":
    main()
