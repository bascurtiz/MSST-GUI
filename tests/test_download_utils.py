"""
tests/test_download_utils.py
Headless tests for backend.download_utils.stream_download.

Run with:
    $env:QT_QPA_PLATFORM="offscreen"; python -m pytest tests/test_download_utils.py -v
No Qt widgets are created, so this runs in any environment.
"""
import os
import re
import threading
import http.server
import functools

import pytest

from backend.download_utils import stream_download

TEST_DATA = bytes((i * 7 + 3) % 256 for i in range(1024 * 1024))  # 1 MiB deterministic


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        rng = self.headers.get("Range")
        data = TEST_DATA
        if rng:
            m = re.match(r"bytes=(\d+)-(\d*)$", rng)
            start = int(m.group(1))
            end = int(m.group(2)) if m.group(2) else len(data) - 1
            end = min(end, len(data) - 1)
            chunk = data[start : end + 1]
            self.send_response(206)
            self.send_header("Content-Range", f"bytes {start}-{end}/{len(data)}")
            self.send_header("Content-Length", str(len(chunk)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(chunk)
        else:
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            self.wfile.write(data)

    def log_message(self, *args):
        pass


class _SlowHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", str(len(TEST_DATA)))
        self.end_headers()
        self.wfile.flush()
        # Stall far longer than the client read timeout.
        import time
        time.sleep(5)

    def log_message(self, *args):
        pass


def _serve(handler_cls):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, srv.server_address[1]


def _tmp(dest):
    d = os.path.join(os.path.dirname(__file__), "_dl_tmp")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, dest)


def test_full_download():
    srv, port = _serve(_RangeHandler)
    try:
        dest = _tmp("full.bin")
        if os.path.exists(dest):
            os.remove(dest)
        ok, msg = stream_download(
            f"http://127.0.0.1:{port}/file", dest, timeout=(5, 5)
        )
        assert ok, msg
        with open(dest, "rb") as f:
            assert f.read() == TEST_DATA
    finally:
        srv.shutdown()
        if os.path.exists(dest):
            os.remove(dest)


def test_resume_from_partial():
    srv, port = _serve(_RangeHandler)
    try:
        dest = _tmp("resume.bin")
        if os.path.exists(dest):
            os.remove(dest)
        # Pre-write the first 40% of the file.
        cut = len(TEST_DATA) * 40 // 100
        with open(dest, "wb") as f:
            f.write(TEST_DATA[:cut])
        ok, msg = stream_download(
            f"http://127.0.0.1:{port}/file", dest, timeout=(5, 5)
        )
        assert ok, msg
        with open(dest, "rb") as f:
            assert f.read() == TEST_DATA
    finally:
        srv.shutdown()
        if os.path.exists(dest):
            os.remove(dest)


def test_cancel_mid_flight():
    srv, port = _serve(_RangeHandler)
    try:
        dest = _tmp("cancel.bin")
        if os.path.exists(dest):
            os.remove(dest)

        state = {"cancel": False}
        # Flip the cancel flag after the first progress callback fires.
        def on_progress(name, cur, total):
            state["cancel"] = True

        ok, msg = stream_download(
            f"http://127.0.0.1:{port}/file", dest,
            progress_callback=on_progress,
            should_cancel=lambda: state["cancel"],
            timeout=(5, 5),
            chunk_size=65536,
        )
        assert ok is False
        assert msg == "cancelled"
        # No crash / no exception propagated.
    finally:
        srv.shutdown()
        if os.path.exists(dest):
            os.remove(dest)


def test_read_timeout():
    srv, port = _serve(_SlowHandler)
    try:
        dest = _tmp("slow.bin")
        if os.path.exists(dest):
            os.remove(dest)
        ok, msg = stream_download(
            f"http://127.0.0.1:{port}/file", dest, timeout=(2, 2)
        )
        assert ok is False
        assert "fail" in msg.lower() or "timed" in msg.lower()
    finally:
        srv.shutdown()
        if os.path.exists(dest):
            os.remove(dest)


def test_no_resume_when_disabled():
    # When resume=False but a partial file exists, the helper should restart
    # from scratch and still produce the full content.
    srv, port = _serve(_RangeHandler)
    try:
        dest = _tmp("noresume.bin")
        if os.path.exists(dest):
            os.remove(dest)
        with open(dest, "wb") as f:
            f.write(b"GARBAGE")
        ok, msg = stream_download(
            f"http://127.0.0.1:{port}/file", dest,
            timeout=(5, 5), resume=False,
        )
        assert ok, msg
        with open(dest, "rb") as f:
            assert f.read() == TEST_DATA
    finally:
        srv.shutdown()
        if os.path.exists(dest):
            os.remove(dest)
