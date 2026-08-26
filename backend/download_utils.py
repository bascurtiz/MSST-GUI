"""
backend/download_utils.py
Robust streaming download helper shared by all model-download code paths.

Features:
  * A single requests.Session with automatic retry/back-off on connection
    errors and 5xx responses (connection pooling / keep-alive reuse).
  * A (connect, read) timeout so a stalled read fails instead of hanging
    the whole process.
  * HTTP Range resume: if a partial file exists and the server advertises
    Accept-Ranges, the download continues from the downloaded offset instead
    of restarting from zero (this is what makes large-model downloads survive
    a dropped connection).
  * **Parallel segmented download**: large files are split into N byte ranges
    and downloaded concurrently across N connections, bypassing per-connection
    CDN throttling (like IDM).
  * Cooperative cancellation via a should_cancel callback checked between
    chunks; the caller decides whether to keep or delete the partial file.
  * Throttled progress callbacks to cut cross-thread signal overhead.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


SEGMENT_SIZE = 8 * 1024 * 1024  # 8 MB per segment
MIN_SEGMENTS = 3                # at least 3 segments to bother
MAX_SEGMENTS = 16               # cap for sanity


def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        connect=3,
        read=3,
        backoff_factor=0.5,
        status_forcelist=(500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def stream_download(
    url: str,
    dest: str,
    *,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    status_callback: Optional[Callable[[str], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    speed_callback: Optional[Callable[[float], None]] = None,
    chunk_size: int = 65536,
    timeout: tuple = (30, 30),
    resume: bool = True,
    throttle: float = 0.1,
    session: Optional[requests.Session] = None,
) -> tuple[bool, str]:
    """Stream *url* to *dest*.

    Returns (ok, message). The caller is responsible for deleting the partial
    file on user-cancellation; this helper keeps the partial so it can be
    resumed on the next attempt or by the caller.
    """
    filename = os.path.basename(dest)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)

    own_session = session is None
    sess = session or _make_session()

    def _cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    def _emit_progress(downloaded: int, total: int) -> None:
        if progress_callback:
            if total > 0:
                progress_callback(filename, downloaded, total)
            else:
                progress_callback(filename, downloaded, downloaded)

    max_attempts = 6
    attempt = 0

    # Offset we are resuming from (0 = fresh download, write mode).
    existing = 0
    if resume and os.path.isfile(dest):
        existing = os.path.getsize(dest)

    try:
        while attempt < max_attempts:
            if _cancelled():
                return False, "cancelled"
            attempt += 1

            req_headers: dict = {}
            if existing > 0 and resume:
                req_headers["Range"] = f"bytes={existing}-"

            try:
                resp = sess.get(url, stream=True, timeout=timeout, headers=req_headers)
            except requests.exceptions.RequestException as e:
                if attempt >= max_attempts:
                    return False, f"Connection failed: {e}"
                time.sleep(min(0.5 * attempt, 4))
                continue

            if resp.status_code == 416:
                # Requested range not satisfiable -> assume already complete.
                resp.close()
                return True, "already complete"

            try:
                resp.raise_for_status()
            except requests.exceptions.HTTPError as e:
                resp.close()
                return False, f"HTTP error: {e}"

            is_partial = resp.status_code == 206

            if is_partial:
                content_range = resp.headers.get("Content-Range", "")
                total = 0
                if "/" in content_range:
                    try:
                        total = int(content_range.split("/")[-1])
                    except ValueError:
                        total = 0
                mode = "ab"
            else:
                try:
                    total = int(resp.headers.get("content-length", 0))
                except (TypeError, ValueError):
                    total = 0
                # Server ignored our Range request -> restart from scratch.
                if existing > 0 and resume:
                    existing = 0
                mode = "wb"

            if status_callback:
                status_callback("Downloading...")

            downloaded = existing
            last_emit = 0.0
            speed_last_dl = existing
            speed_last_ts = time.time()
            wrote_any = False

            try:
                with open(dest, mode) as f:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if _cancelled():
                            return False, "cancelled"
                        if chunk:
                            f.write(chunk)
                            wrote_any = True
                            downloaded += len(chunk)
                            now = time.time()
                            if now - last_emit >= throttle:
                                _emit_progress(downloaded, total)
                                last_emit = now
                            if speed_callback and now - speed_last_ts >= 0.5:
                                delta = downloaded - speed_last_dl
                                if delta > 0:
                                    mbps = (delta / (now - speed_last_ts)) * 8 / 1024 / 1024
                                    speed_callback(mbps)
                                speed_last_dl = downloaded
                                speed_last_ts = now
                _emit_progress(downloaded, total)
            except requests.exceptions.RequestException:
                # Connection broke mid-stream. Resume if the server supports
                # Range; otherwise a non-resumable source cannot be completed.
                if wrote_any:
                    existing = os.path.getsize(dest)
                    if is_partial:
                        continue
                return False, "Download failed (connection dropped)"
            finally:
                resp.close()

            # Connection closed cleanly but short of the full size.
            if total > 0 and downloaded < total:
                if is_partial:
                    existing = downloaded
                    continue
                return False, "Download incomplete (server does not support resume)"

            return True, "ok"

        return False, "Download failed after retries"
    finally:
        if own_session:
            sess.close()


def _head_file_info(url: str, session: requests.Session, timeout: tuple = (15, 15)) -> tuple[int, bool]:
    """Return (content_length, accept_ranges) via HEAD."""
    try:
        resp = session.head(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        cl = int(resp.headers.get("Content-Length", 0))
        accept_ranges = "bytes" in resp.headers.get("Accept-Ranges", "")
        return cl, accept_ranges
    except Exception:
        return 0, False


def parallel_download(
    url: str,
    dest: str,
    *,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
    speed_callback: Optional[Callable[[float], None]] = None,
    should_cancel: Optional[Callable[[], bool]] = None,
    min_segment_size: int = 5 * 1024 * 1024,
    timeout: tuple = (60, 60),
    session: Optional[requests.Session] = None,
) -> tuple[bool, str]:
    """Download *url* using multiple parallel HTTP Range requests.

    For large files behind CDNs that throttle per-connection throughput,
    this splits the file into *N* segments and downloads them concurrently,
    then concatenates the parts (like IDM).

    Falls back to ``stream_download`` for small files or servers that do
    not support Range requests.
    """
    filename = os.path.basename(dest)
    parent = os.path.dirname(dest)
    if parent:
        os.makedirs(parent, exist_ok=True)

    own_session = session is None
    sess = session or _make_session()

    file_size, accept_ranges = _head_file_info(url, sess, timeout)
    if file_size < min_segment_size or not accept_ranges:
        if own_session:
            sess.close()
            return stream_download(url, dest, progress_callback=progress_callback,
                                   speed_callback=speed_callback,
                                   should_cancel=should_cancel, timeout=timeout,
                                   chunk_size=1048576, session=None)
        return stream_download(url, dest, progress_callback=progress_callback,
                               speed_callback=speed_callback,
                               should_cancel=should_cancel, timeout=timeout,
                               chunk_size=1048576, session=sess)

    num = max(MIN_SEGMENTS, min(MAX_SEGMENTS, file_size // SEGMENT_SIZE))
    seg_size = file_size // num
    ranges = [(i * seg_size, (i + 1) * seg_size - 1 if i < num - 1 else file_size - 1)
              for i in range(num)]

    def _cancelled() -> bool:
        return bool(should_cancel and should_cancel())

    seg_progress: list[int] = [0] * num
    seg_ok: list[bool] = [False] * num
    seg_msg: list[str] = [""] * num
    lock = threading.Lock()

    def _download_seg(idx: int, start: int, end: int):
        part_path = f"{dest}.part.{idx}"
        attempt = 0
        max_attempts = 4
        while attempt < max_attempts and not _cancelled():
            attempt += 1
            try:
                headers = {"Range": f"bytes={start}-{end}"}
                resp = sess.get(url, stream=True, timeout=timeout, headers=headers)
                if resp.status_code == 416:
                    resp.close()
                    seg_ok[idx] = True
                    seg_progress[idx] = end - start + 1
                    return
                resp.raise_for_status()
                with open(part_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1048576):
                        if _cancelled():
                            resp.close()
                            return
                        if chunk:
                            f.write(chunk)
                            with lock:
                                seg_progress[idx] += len(chunk)
                resp.close()
                if os.path.getsize(part_path) >= end - start + 1:
                    seg_ok[idx] = True
                    seg_msg[idx] = ""
                    return
            except requests.exceptions.RequestException:
                if attempt >= max_attempts:
                    seg_msg[idx] = f"segment {idx} failed"
                    return
                time.sleep(min(0.5 * attempt, 3))
        if _cancelled():
            seg_msg[idx] = "cancelled"
        else:
            seg_msg[idx] = f"segment {idx} failed after retries"

    threads = []
    for i, (s, e) in enumerate(ranges):
        t = threading.Thread(target=_download_seg, args=(i, s, e))
        t.start()
        threads.append(t)

    _stop_monitor = False

    def _monitor():
        last_dl = 0
        last_ts = time.time()
        while not _stop_monitor and not _cancelled():
            time.sleep(0.25)
            with lock:
                total = sum(seg_progress)
            if progress_callback:
                progress_callback(filename, total, file_size)
            now = time.time()
            elapsed = now - last_ts
            if elapsed >= 0.5:
                delta = total - last_dl
                if delta > 0 and elapsed > 0:
                    mbps = (delta / elapsed) * 8 / 1024 / 1024
                    if speed_callback:
                        speed_callback(mbps)
                last_dl = total
                last_ts = now

    monitor = threading.Thread(target=_monitor, daemon=True)
    monitor.start()

    for t in threads:
        t.join()

    _stop_monitor = True
    monitor.join(timeout=1)

    if _cancelled():
        for i in range(num):
            p = f"{dest}.part.{i}"
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
        if own_session:
            sess.close()
        return False, "cancelled"

    for i in range(num):
        if not seg_ok[i]:
            for j in range(num):
                p = f"{dest}.part.{j}"
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            if own_session:
                sess.close()
            return False, seg_msg[i] or f"segment {i} incomplete"

    try:
        with open(dest, "wb") as out:
            for i in range(num):
                part_path = f"{dest}.part.{i}"
                with open(part_path, "rb") as f:
                    while True:
                        buf = f.read(1048576)
                        if not buf:
                            break
                        out.write(buf)
                os.remove(part_path)
    except OSError as e:
        if own_session:
            sess.close()
        return False, f"reassembly failed: {e}"

    if progress_callback:
        progress_callback(filename, file_size, file_size)
    if own_session:
        sess.close()
    return True, "ok"
