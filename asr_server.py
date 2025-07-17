#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Light-weight Sherpa-ONNX HTTP ASR server (thread-safe).
"""

from __future__ import annotations
import socket
import argparse
import io
import json
import logging
import os
import sys
import time
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Final, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Logging                                                                     #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=os.getenv("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s.%(msecs)03d [%(levelname)8s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("ASR")

# --------------------------------------------------------------------------- #
# Constants                                                                   #
# --------------------------------------------------------------------------- #
EXPECTED_RATE:     Final[int] = 16_000
EXPECTED_CHANNELS: Final[int] = 1
EXPECTED_WIDTH_B:  Final[int] = 2           # 16-bit PCM
TAIL_PADDING:      Final[np.ndarray] = np.zeros(int(0.5 * EXPECTED_RATE), np.float32)

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    base_path = sys._MEIPASS  # For packaged apps
else:
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))  # For normal scripts
    except NameError:
        base_path = os.path.abspath(".")  # Fallback

MODEL_DIR = os.path.join(base_path, "assets", "sensevoicesmallonnx")
MODEL_PATH   = os.path.join(MODEL_DIR, "model.onnx")
TOKENS_PATH  = os.path.join(MODEL_DIR, "tokens.txt")

# --------------------------------------------------------------------------- #
# Model loading                                                               #
# --------------------------------------------------------------------------- #
try:
    import sherpa_onnx
except ImportError as exc:          # pragma: no cover
    log.critical("sherpa_onnx not found – install it via `pip install sherpa-onnx`")
    raise SystemExit(1) from exc


def load_recognizer() -> "sherpa_onnx.OfflineRecognizer":
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(TOKENS_PATH)):
        log.critical("Model assets missing under %s", MODEL_DIR)
        raise SystemExit(1)

    ts0 = time.perf_counter()
    log.info("Begin loading model...")
    rec = sherpa_onnx.OfflineRecognizer.from_sense_voice(
        model=MODEL_PATH,
        tokens=TOKENS_PATH,
        language="",              # auto
        use_itn=True,
        num_threads=max(1, os.cpu_count() // 2),
        provider="cpu",
        debug=False,
    )
    cost = time.perf_counter() - ts0
    log.info("Model loaded in %.2f s", cost)
    return rec


RECOGNIZER = load_recognizer()  # single global instance, thread-safe


# --------------------------------------------------------------------------- #
# Core ASR function                                                           #
# --------------------------------------------------------------------------- #
def transcribe_pcm16le(pcm_bytes: bytes) -> str:
    """
    Parameters
    ----------
    pcm_bytes : bytes
        16-bit little endian, mono, 16 kHz PCM audio.

    Returns
    -------
    str
        Transcribed text.
    """
    ts0 = time.perf_counter()

    # -- Zero-copy int16 view → float32 normalised --------------------------- #
    audio_i16 = np.frombuffer(memoryview(pcm_bytes), dtype=np.int16)
    audio_f32 = audio_i16.astype(np.float32) / 32768.0

    stream = RECOGNIZER.create_stream()
    stream.accept_waveform(EXPECTED_RATE, audio_f32)
    stream.accept_waveform(EXPECTED_RATE, TAIL_PADDING)

    RECOGNIZER.decode_stream(stream)
    text = stream.result.text

    cost = time.perf_counter() - ts0
    log.info("ASR processed %d samples in %.2f s, result: %s", audio_f32.shape[0], cost, text)
    log.debug("Transcribed %d samples in %.2f s → %s",
              audio_f32.shape[0], cost, text)
    return text


# --------------------------------------------------------------------------- #
# HTTP handler                                                                #
# --------------------------------------------------------------------------- #
class ASRHandler(BaseHTTPRequestHandler):
    server_version = "SherpaASR/1.0"

    def log_message(self, fmt, *args):                                        # noqa: N802
        log.info("%s – " + fmt, self.address_string(), *args)

    def _json(self, code: int, payload: dict):
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _bad_request(self, message: str):
        self._json(400, {"status": "error", "message": message})

    def do_GET(self):                                                         # noqa: N802
        if self.path != "/":
            self._json(404, {"status": "error", "message": "Not Found"})
            return

        self._json(
            200,
            {
                "status": "ok",
                "model_loaded": True,
                "usage": "POST a 16 kHz / 16-bit / mono WAV file to /asr",
            },
        )

    def do_POST(self):                                                        # noqa: N802
        if self.path != "/asr":
            self._json(404, {"status": "error", "message": "Not Found"})
            return

        length = self.headers.get("Content-Length")
        if length is None:
            return self._bad_request("Missing Content-Length header")

        try:
            body = self.rfile.read(int(length))
        except Exception as exc:
            return self._bad_request(f"Failed to read body: {exc}")

        # Parse WAV in-memory
        try:
            with wave.open(io.BytesIO(body), "rb") as wf:
                if (
                    wf.getnchannels() != EXPECTED_CHANNELS
                    or wf.getsampwidth() != EXPECTED_WIDTH_B
                    or wf.getframerate() != EXPECTED_RATE
                ):
                    return self._bad_request(
                        "Audio must be 16 kHz, 16-bit, mono PCM WAV"
                    )
                pcm_bytes = wf.readframes(wf.getnframes())
        except wave.Error as exc:
            return self._bad_request(f"Invalid WAV file: {exc}")

        # Transcribe
        try:
            text = transcribe_pcm16le(pcm_bytes)
        except Exception as exc:                                              # pragma: no cover
            log.exception("ASR failed")
            return self._json(500, {"status": "error", "message": str(exc)})

        self._json(200, {"status": "success", "result": text})

# --------------------------------------------------------------------------- #
# Dual-protocol server                                                        #
# --------------------------------------------------------------------------- #
class DualStackServer(ThreadingHTTPServer):
    """HTTP server that supports both IPv4 and IPv6 simultaneously."""
    
    def __init__(self, server_address: Tuple[str, int], RequestHandlerClass, ipv4: bool = True, ipv6: bool = True):
        self.ipv4 = ipv4
        self.ipv6 = ipv6
        self.address_family = socket.AF_INET6 if ipv6 else socket.AF_INET
        ThreadingHTTPServer.__init__(self, server_address, RequestHandlerClass)
        
    def server_bind(self):
        if self.ipv6:
            try:
                self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
            except Exception as e:
                log.warning("Could not enable dual-stack mode (IPV6_V6ONLY=0): %s", e)
        
        ThreadingHTTPServer.server_bind(self)

# --------------------------------------------------------------------------- #
# Network utilities                                                           #
# --------------------------------------------------------------------------- #
def get_network_interfaces(ipv6: bool = False) -> List[str]:
    addresses = set()
    family = socket.AF_INET6 if ipv6 else socket.AF_INET

    try:
        for iface in socket.getaddrinfo(socket.gethostname(), None):
            if iface[0] == family:
                addr = iface[4][0]
                if ipv6:
                    if not (addr.startswith("fe80") or addr == "::1"):
                        addresses.add(addr)
                else:
                    if not addr.startswith("127."):
                        addresses.add(addr)
    except Exception:
        pass

    try:
        with socket.socket(family, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80) if not ipv6 else ("2001:4860:4860::8888", 80))
            addr = s.getsockname()[0]
            if ipv6:
                if not (addr.startswith("fe80") or addr == "::1"):
                    addresses.add(addr)
            else:
                if not addr.startswith("127."):
                    addresses.add(addr)
    except Exception:
        pass

    return sorted(addresses)

# --------------------------------------------------------------------------- #
# Entrypoint                                                                  #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Sherpa-ONNX ASR HTTP server")
    parser.add_argument("--host", default=None, 
                      help="Specific host to bind to (e.g., '0.0.0.0', '::', 'localhost')")
    parser.add_argument("--port", default=7860, type=int, help="Port to listen on")
    parser.add_argument("--ip-version", choices=["4", "6", "dual"], default="dual",
                      help="IP version to use: 4=IPv4 only, 6=IPv6 only, dual=both")
    parser.add_argument("--scope", choices=["local", "all"], default="all",
                      help="Binding scope: local=loopback only, all=all interfaces")
    args = parser.parse_args()
    
    ipv4 = args.ip_version in ["4", "dual"]
    ipv6 = args.ip_version in ["6", "dual"]

    if args.host:
        bind_host = args.host
        if ":" in bind_host and not ipv6:
            log.warning("IPv6 host specified but IPv6 is disabled")
        bind_all = bind_host in ("0.0.0.0", "::", "")
    else:
        if args.scope == "local":
            bind_host = "::1" if ipv6 else "127.0.0.1"
            bind_all = False
        else:
            bind_host = "::" if ipv6 else "0.0.0.0"
            bind_all = True

    try:
        server = DualStackServer(
            server_address=(bind_host, args.port),
            RequestHandlerClass=ASRHandler,
            ipv4=ipv4,
            ipv6=ipv6
        )
    except OSError as e:
        log.critical("Failed to start server: %s", e)
        sys.exit(1)
    
    log.info("Server started on port %d", args.port)
    log.info("Protocols: IPv4=%s, IPv6=%s", ipv4, ipv6)
    
    if bind_all:
        if ipv4:
            for addr in get_network_interfaces(ipv6=False):
                log.info("IPv4: http://%s:%d/", addr, args.port)
        if ipv6:
            for addr in get_network_interfaces(ipv6=True):
                log.info("IPv6: http://[%s]:%d/", addr, args.port)
    else:
        if ":" in bind_host:
            log.info("Listening on: http://[%s]:%d/", bind_host, args.port)
        else:
            log.info("Listening on: http://%s:%d/", bind_host, args.port)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Shutting down...")
    finally:
        server.server_close()

if __name__ == "__main__":
    main()
