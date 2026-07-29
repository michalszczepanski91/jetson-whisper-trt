#!/usr/bin/env python3
"""Compare transcription latency and memory across whisper / whisper_trt
(/ faster_whisper if installed) on the same audio file.

Logic adapted from NVIDIA-AI-IOT/whisper_trt's examples/profile_backend.py
(MIT licensed) https://github.com/NVIDIA-AI-IOT/whisper_trt — restructured
into the subprocess-per-backend comparison table style used by
jetson-yolov8-trt/scripts/benchmark_compare.py (clean process memory
isolation between backends).

Usage (inside dev container):
    python scripts/benchmark.py --audio speech.wav --model base.en
"""

import argparse
import json
import subprocess
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model", choices=["tiny.en", "base.en", "small.en"], default="base.en")
    p.add_argument("--iters", type=int, default=3)
    # subprocess-only args
    p.add_argument("--_backend", default=None)
    return p.parse_args()


def subprocess_mode(args):
    """Single-backend profile — called from subprocess."""
    import os
    import time
    import psutil
    import torch

    def process_memory():
        return psutil.Process(os.getpid()).memory_info().rss

    def load(backend, model_name):
        if backend == "whisper_trt":
            from whisper_trt import load_trt_model
            return load_trt_model(model_name), False
        elif backend == "whisper":
            from whisper import load_model
            return load_model(model_name), False
        elif backend == "faster_whisper":
            from faster_whisper import WhisperModel
            return WhisperModel(model_name), True
        raise ValueError(backend)

    def transcribe(model, is_faster_whisper, audio):
        result = model.transcribe(audio)
        if is_faster_whisper:
            segs, _info = result
            result = {"text": "".join(seg.text for seg in segs)}
        return result

    start_mem = process_memory()
    model, is_faster_whisper = load(args._backend, args.model)

    # warmup (also covers whisper_trt's first-call TensorRT engine build, which
    # must not be counted as steady-state latency)
    transcribe(model, is_faster_whisper, args.audio)

    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        result = transcribe(model, is_faster_whisper, args.audio)
    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()
    latency_s = (time.perf_counter() - t0) / args.iters

    end_mem = process_memory()

    print(json.dumps({
        "backend": args._backend,
        "text": result["text"],
        "latency_s": latency_s,
        "mem_mb": (end_mem - start_mem) >> 20,
    }))


def run_in_subprocess(base_args, backend):
    cmd = [
        sys.executable, __file__,
        "--audio", base_args.audio,
        "--model", base_args.model,
        "--iters", str(base_args.iters),
        "--_backend", backend,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            for line in reversed(result.stdout.strip().splitlines()):
                try:
                    return json.loads(line)
                except json.JSONDecodeError:
                    continue
        else:
            stderr = result.stderr.strip().splitlines()
            print(f"  failed: {stderr[-1] if stderr else 'unknown error'}")
        return None
    except subprocess.TimeoutExpired:
        print("  timed out")
        return None


def main():
    args = parse_args()

    if args._backend is not None:
        subprocess_mode(args)
        return

    results = []
    for backend in ("whisper", "whisper_trt", "faster_whisper"):
        print(f"Benchmarking {backend}...")
        r = run_in_subprocess(args, backend)
        if r:
            results.append(r)
        elif backend == "faster_whisper":
            print("  (skipped — faster_whisper not installed; see README to add it)")

    if not results:
        print("No backends produced results.")
        return

    print()
    print(f"{'Backend':<16} {'Latency(s)':>12} {'Memory(MB)':>12}")
    print("-" * 42)
    for r in results:
        print(f"{r['backend']:<16} {r['latency_s']:>12.3f} {r['mem_mb']:>12d}")

    baseline = next((r["latency_s"] for r in results if r["backend"] == "whisper"), None)
    if baseline:
        print()
        print("Speedup vs whisper (PyTorch) baseline:")
        for r in results:
            if r["backend"] != "whisper":
                print(f"  {r['backend']:<16} {baseline / r['latency_s']:.1f}x faster")


if __name__ == "__main__":
    main()
