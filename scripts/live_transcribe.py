#!/usr/bin/env python3
"""Live mic -> Silero VAD -> WhisperTRT transcription loop.

Single-process, single-channel version for this project's hardware (Jabra
EVOLVE 30 II, confirmed in Phase 0 as the only mic that captures native mono
16kHz without resampling — see embedded-ai-chain's docs/environment.md).
Structurally adapted from NVIDIA-AI-IOT/whisper_trt's
examples/live_transcription.py (MIT licensed)
https://github.com/NVIDIA-AI-IOT/whisper_trt, which targets a 6-channel
ReSpeaker array via a multiprocess pipeline — simplified to one process and
one channel since that's the actual microphone this robot has.

Usage (inside dev container):
    python scripts/live_transcribe.py --config configs/pipeline.yaml
"""

import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyaudio
import yaml


class BridgeClient:
    """Sends each transcript to embedded-ai-chain's host-side SttBridge
    (src/stt_bridge.py) over a plain TCP socket — this container's
    docker-compose.yml uses network_mode: host, so 127.0.0.1 on the host is
    directly reachable, no port mapping needed.

    Lazily connects and reconnects on failure rather than raising: the host
    orchestrator may not be up yet (or may restart), and a live mic session
    should keep running (and keep printing/logging locally) regardless of
    whether anything is listening on the other end.
    """

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._sock: socket.socket | None = None

    def send(self, text: str, transcribe_seconds: float) -> None:
        line = (json.dumps({"text": text, "transcribe_seconds": transcribe_seconds}) + "\n").encode()
        for attempt in range(2):  # one retry after a fresh reconnect
            try:
                if self._sock is None:
                    self._sock = socket.create_connection((self._host, self._port), timeout=1.0)
                self._sock.sendall(line)
                return
            except OSError as exc:
                print(f"  (bridge send failed, will retry connecting: {exc})")
                if self._sock is not None:
                    try:
                        self._sock.close()
                    except OSError:
                        pass
                self._sock = None
                if attempt == 1:
                    print("  (bridge unreachable, continuing without it)")

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None


def find_device_index(name_contains: str) -> int:
    p = pyaudio.PyAudio()
    try:
        info = p.get_host_api_info_by_index(0)
        for i in range(info.get("deviceCount")):
            dev = p.get_device_info_by_host_api_device_index(0, i)
            if name_contains.lower() in dev.get("name", "").lower() and dev.get("maxInputChannels", 0) > 0:
                return i
    finally:
        p.terminate()
    raise RuntimeError(f"No input device matching {name_contains!r} found (run `make list-devices`)")


def audio_chunk_to_normalized(raw_bytes: bytes) -> np.ndarray:
    audio = np.frombuffer(raw_bytes, dtype=np.int16)
    return audio.astype(np.float32) / 32768.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pipeline.yaml")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    audio_cfg = cfg["audio"]
    model_cfg = cfg["model"]
    vad_cfg = cfg["vad"]
    output_cfg = cfg["output"]
    bridge_cfg = cfg.get("bridge", {})

    sample_rate = audio_cfg["sample_rate"]
    chunk_size = audio_cfg["chunk_size"]

    device_index = audio_cfg.get("device_index")
    if device_index is None:
        device_index = find_device_index(audio_cfg["device_name_contains"])

    print(f"Loading Silero VAD and {model_cfg['size']} WhisperTRT model "
          f"(first run builds+caches the TensorRT engine, can take a while)...")
    from whisper_trt import load_trt_model
    from whisper_trt.vad import load_vad

    vad = load_vad()
    vad(np.zeros(chunk_size, dtype=np.float32), sr=sample_rate)  # warmup

    model = load_trt_model(model_cfg["size"])
    model.transcribe(np.zeros(chunk_size, dtype=np.float32))  # warmup

    out_fh = None
    if output_cfg["mode"] == "file":
        out_fh = open(output_cfg["file_path"], "a")

    bridge = None
    if bridge_cfg.get("enabled", False):
        bridge = BridgeClient(bridge_cfg.get("host", "127.0.0.1"), bridge_cfg.get("port", 8765))

    def emit(text: str, transcribe_s: float):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] ({transcribe_s:.2f}s) {text}"
        print(line)
        if out_fh:
            out_fh.write(line + "\n")
            out_fh.flush()
        if bridge:
            bridge.send(text, transcribe_s)

    pa = pyaudio.PyAudio()
    stream = pa.open(
        rate=sample_rate,
        format=pa.get_format_from_width(2),
        channels=audio_cfg["channels"],
        input=True,
        input_device_index=device_index,
        frames_per_buffer=chunk_size,
    )

    threshold = vad_cfg["speech_threshold"]
    window = deque(maxlen=vad_cfg["max_filter_window"])
    speech_buffer = []
    prev_is_voice = False

    print(f"Listening on device #{device_index}. Ctrl-C to stop.")
    try:
        while True:
            raw = stream.read(chunk_size, exception_on_overflow=False)
            chunk = audio_chunk_to_normalized(raw)

            voice_prob = float(vad(chunk, sr=sample_rate).flatten()[0])
            window.append((chunk, voice_prob))
            is_voice = any(prob > threshold for _, prob in window)

            if is_voice and not prev_is_voice:
                speech_buffer = [c for c, _ in window]
            elif is_voice:
                speech_buffer.append(chunk)
            elif prev_is_voice and not is_voice:
                segment = np.concatenate(speech_buffer)
                t0 = time.perf_counter()
                result = model.transcribe(segment)
                emit(result["text"], time.perf_counter() - t0)
                speech_buffer = []

            prev_is_voice = is_voice
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()
        if out_fh:
            out_fh.close()
        if bridge:
            bridge.close()


if __name__ == "__main__":
    sys.exit(main())
