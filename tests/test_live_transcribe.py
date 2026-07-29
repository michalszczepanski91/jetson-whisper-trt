"""Unit tests for BridgeClient (scripts/live_transcribe.py) — the socket
client side of the host bridge (embedded-ai-chain's src/stt_bridge.py).
The mic/VAD/whisper_trt parts of live_transcribe.py need real hardware and
aren't unit-testable; this covers the one piece that is: does it send the
right bytes, and does it degrade gracefully instead of crashing when nothing
is listening (the host orchestrator may not be up yet, or may restart).

Run with: python -m pytest tests/ (inside the dev container, or anywhere
with pyaudio/numpy/yaml importable — those are only imported at module load
by live_transcribe.py, so this needs the same deps as the live pipeline).
"""

import json
import socket
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from live_transcribe import BridgeClient  # noqa: E402


def _recv_line(server_sock, timeout=2):
    server_sock.settimeout(timeout)
    conn, _addr = server_sock.accept()
    with conn:
        conn.settimeout(timeout)
        data = b""
        while b"\n" not in data:
            chunk = conn.recv(4096)
            if not chunk:
                break
            data += chunk
        return json.loads(data.split(b"\n", 1)[0])


def test_send_delivers_correct_json():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    result = {}

    def accept():
        result["data"] = _recv_line(server)

    t = threading.Thread(target=accept)
    t.start()

    client = BridgeClient("127.0.0.1", port)
    client.send("test utterance", 1.23)
    t.join(timeout=2)
    client.close()
    server.close()

    assert result["data"] == {"text": "test utterance", "transcribe_seconds": 1.23}


def test_send_to_nothing_listening_does_not_raise():
    # Nothing bound on this port - send() must degrade gracefully, not raise,
    # since a live mic session should keep running even if the host bridge
    # isn't up.
    client = BridgeClient("127.0.0.1", 1)  # port 1 - reserved, nothing listens
    client.send("should not crash", 0.5)  # must not raise
    client.close()


def test_close_without_ever_connecting_does_not_raise():
    client = BridgeClient("127.0.0.1", 8765)
    client.close()  # never sent anything, never connected
