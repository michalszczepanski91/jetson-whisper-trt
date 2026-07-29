#!/usr/bin/env python3
"""
Validate that all required packages and hardware capabilities are present.
Run this inside the Docker container.

Usage:
    python scripts/validate_install.py
"""

import sys
import subprocess

RESET  = "\033[0m"
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"


def check(label: str, fn):
    try:
        result = fn()
        print(f"  {GREEN}✔{RESET}  {label}: {result}")
        return True
    except Exception as e:
        print(f"  {RED}✘{RESET}  {label}: {RED}{e}{RESET}")
        return False


def main():
    print(f"\n{BOLD}═══ Jetson WhisperTRT — Install Validator ═══{RESET}\n")
    results = []

    print(f"{BOLD}[ Python ]{RESET}")
    results.append(check("Python version",
        lambda: sys.version.split()[0]))

    print(f"\n{BOLD}[ CUDA / GPU ]{RESET}")
    def cuda_available():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        return "yes"
    results.append(check("torch.cuda available", cuda_available))
    results.append(check("CUDA device name",
        lambda: __import__("torch").cuda.get_device_name(0)))
    results.append(check("CUDA version",
        lambda: __import__("torch").version.cuda))

    print(f"\n{BOLD}[ Core Libraries ]{RESET}")
    results.append(check("torch",
        lambda: __import__("torch").__version__))
    results.append(check("numpy",
        lambda: __import__("numpy").__version__))
    results.append(check("PyYAML",
        lambda: __import__("yaml").__version__))
    results.append(check("psutil",
        lambda: __import__("psutil").__version__))

    print(f"\n{BOLD}[ TensorRT / torch2trt ]{RESET}")
    results.append(check("tensorrt",
        lambda: __import__("tensorrt").__version__))
    results.append(check("torch2trt",
        lambda: __import__("torch2trt").__file__))

    print(f"\n{BOLD}[ ASR ]{RESET}")
    results.append(check("openai-whisper",
        lambda: __import__("whisper").__file__ and "importable"))
    results.append(check("whisper_trt",
        lambda: __import__("whisper_trt").__version__))
    results.append(check("onnxruntime (Silero VAD backend)",
        lambda: __import__("onnxruntime").__version__))

    print(f"\n{BOLD}[ Audio ]{RESET}")
    def pyaudio_devices():
        import pyaudio
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        n = info.get("deviceCount")
        names = [p.get_device_info_by_host_api_device_index(0, i).get("name") for i in range(n)]
        p.terminate()
        return f"{n} device(s): {names}"
    results.append(check("pyaudio device enumeration", pyaudio_devices))

    print(f"\n{BOLD}[ Jetson System ]{RESET}")
    def jetpack_version():
        out = subprocess.check_output(
            ["cat", "/etc/nv_tegra_release"], text=True)
        return out.split("\n")[0].strip()
    results.append(check("JetPack release", jetpack_version))

    def disk_free():
        out = subprocess.check_output(["df", "-h", "/"], text=True)
        line = [l for l in out.splitlines() if "/" in l][-1]
        return line.split()[3] + " free"
    results.append(check("Disk space", disk_free))

    # ── Summary ───────────────────────────────────────────────────────────────
    passed = sum(results)
    total  = len(results)
    colour = GREEN if passed == total else (YELLOW if passed > total * 0.7 else RED)
    status = "ALL CHECKS PASSED" if passed == total else f"{total - passed} CHECK(S) FAILED"
    print(f"\n{BOLD}Result: {colour}{passed}/{total} — {status}{RESET}\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
