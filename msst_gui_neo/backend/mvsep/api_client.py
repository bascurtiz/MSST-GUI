"""backend/mvsep/api_client.py
MVSep API client with upload/poll/download, retry, timeout, rate limit handling.
Uses the correct MVSep API endpoints:
  - POST /api/separation/create
  - GET  /api/separation/get
All operations are designed to run in background threads.
"""
import os
import time
import requests
import threading
from urllib.parse import urlparse, urlunparse
from typing import Optional, Callable
from datetime import datetime

from backend.mvsep.models import MVSepJob, MVSepModel


MVSEP_API_BASE = "https://mvsep.com/api"
CREATE_ENDPOINT = f"{MVSEP_API_BASE}/separation/create"
STATUS_ENDPOINT = f"{MVSEP_API_BASE}/separation/get"

MAX_RETRIES = 3
RETRY_DELAY = 5
TIMEOUT_UPLOAD = 300
TIMEOUT_POLL = 30
TIMEOUT_DOWNLOAD = 180
POLL_INTERVAL = 10
DOWNLOAD_CHUNK_SIZE = 262144

MIRROR_BASE = "https://mirror.mvsep.com"


class MVSepAPIError(Exception):
    """Base exception for MVSep API errors."""
    pass


class InvalidAPIKeyError(MVSepAPIError):
    """Raised when the API key is invalid."""
    pass


class RateLimitError(MVSepAPIError):
    """Raised when rate limited by the API."""
    pass


class UploadTimeoutError(MVSepAPIError):
    """Raised when upload times out."""
    pass


class ProcessingTimeoutError(MVSepAPIError):
    """Raised when processing takes too long."""
    pass


class MVSepApiClient:
    def __init__(self, api_token: str, log_callback: Optional[Callable[[str], None]] = None):
        self._token = api_token.strip()
        self._log = log_callback or (lambda m: None)
        self._cancel_requested = False
        self._lock = threading.Lock()
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://mvsep.com/",
        })

    def _log_msg(self, msg: str):
        self._log(f"[MVSep] {msg}")

    def cancel(self):
        with self._lock:
            self._cancel_requested = True

    @property
    def is_cancelled(self) -> bool:
        with self._lock:
            return self._cancel_requested

    def _retry_request(self, method, url, max_retries=MAX_RETRIES, **kwargs):
        last_error = None
        for attempt in range(max_retries):
            if self.is_cancelled:
                raise MVSepAPIError("Cancelled by user")
            try:
                resp = self._session.request(method.__name__, url, **kwargs)
                if resp.status_code == 401:
                    raise InvalidAPIKeyError("Invalid MVSep API key")
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", RETRY_DELAY))
                    self._log_msg(f"Rate limited. Waiting {retry_after}s...")
                    time.sleep(retry_after)
                    last_error = RateLimitError("Rate limited")
                    continue
                if resp.status_code == 400:
                    raise MVSepAPIError(f"Bad request: {resp.text}")
                if resp.status_code == 500:
                    wait = RETRY_DELAY * (attempt + 1)
                    self._log_msg(f"Server error. Waiting {wait}s...")
                    time.sleep(wait)
                    last_error = MVSepAPIError("Server error")
                    continue
                if resp.status_code == 503:
                    wait = RETRY_DELAY * (attempt + 1)
                    self._log_msg(f"Service unavailable. Waiting {wait}s...")
                    time.sleep(wait)
                    last_error = MVSepAPIError("Service unavailable")
                    continue
                resp.raise_for_status()
                return resp
            except (requests.RequestException, InvalidAPIKeyError, RateLimitError) as e:
                if isinstance(e, (InvalidAPIKeyError, RateLimitError)):
                    raise
                last_error = e
                if attempt < max_retries - 1:
                    delay = RETRY_DELAY * (2 ** attempt)
                    self._log_msg(f"Request failed ({e}). Retrying in {delay}s...")
                    time.sleep(delay)
        raise last_error or MVSepAPIError("Request failed after retries")

    def upload_file(self, file_path: str, model: MVSepModel) -> MVSepJob:
        self._log_msg(f"Uploading {os.path.basename(file_path)} for {model.value}...")
        job = MVSepJob(task_hash="", model=model, status="uploading")

        with open(file_path, "rb") as f:
            files = {"audiofile": (os.path.basename(file_path), f, "audio/wav")}
            data = {
                "api_token": self._token,
                "sep_type": str(model.sep_type),
                "output_format": "1",
                "is_demo": "0",
            }
            if model.add_opt1:
                data["add_opt1"] = model.add_opt1

            resp = self._retry_request(
                requests.post,
                CREATE_ENDPOINT,
                files=files,
                data=data,
                timeout=TIMEOUT_UPLOAD,
            )

        result = resp.json()
        if not result.get("success"):
            error_msg = result.get("error", "Unknown error")
            raise MVSepAPIError(f"Upload failed: {error_msg}")

        job.task_hash = result.get("data", {}).get("hash", "")
        if not job.task_hash:
            raise MVSepAPIError("No task hash returned from upload")

        job.status = "waiting"
        job.upload_progress = 100.0
        self._log_msg(f"Upload complete. Task hash: {job.task_hash}")
        return job

    def poll_status(self, job: MVSepJob) -> dict:
        if self.is_cancelled:
            raise MVSepAPIError("Cancelled by user")

        params = {"hash": job.task_hash}
        resp = self._retry_request(
            requests.get,
            STATUS_ENDPOINT,
            params=params,
            timeout=TIMEOUT_POLL,
        )

        data = resp.json()
        success = data.get("success", False)

        if success:
            files_data = data.get("data", {}).get("files", [])
            if files_data:
                job.status = "done"
                job.processing_progress = 100.0
                job.completed_at = datetime.now()
                job.download_urls = [f.get("url", "") for f in files_data if f.get("url")]
            else:
                job.status = "processing"
                job.processing_progress = 0.0
        else:
            job.status = "failed"
            job.error = data.get("data", {}).get("error", data.get("error", "Unknown error"))

        return data

    def wait_for_completion(self, job: MVSepJob, timeout: int = 3600, progress_callback: Optional[Callable[[float], None]] = None) -> MVSepJob:
        self._log_msg(f"Waiting for task {job.task_hash} to complete...")
        time.sleep(5)
        start = time.time()
        poll_count = 0

        while time.time() - start < timeout:
            if self.is_cancelled:
                job.status = "cancelled"
                return job

            try:
                data = self.poll_status(job)
                if progress_callback:
                    progress_callback(job.processing_progress)

                if job.is_complete:
                    self._log_msg(f"Task {job.task_hash} completed.")
                    return job

                if job.is_failed:
                    self._log_msg(f"Task {job.task_hash} failed: {job.error}")
                    self._log_msg(f"Raw response: {data}")
                    return job

                poll_count += 1
                if poll_count % 6 == 0:
                    self._log_msg(f"Still processing... ({int(time.time() - start)}s elapsed)")

            except (InvalidAPIKeyError, RateLimitError):
                raise
            except Exception as e:
                self._log_msg(f"Poll error: {e}")

            time.sleep(POLL_INTERVAL)

        job.status = "timeout"
        job.error = f"Processing timed out after {timeout}s"
        raise ProcessingTimeoutError(job.error)

    def _select_instrumental_url(self, urls: list) -> Optional[str]:
        instrumental_keywords = ["instrumental", "inst", "other", "accompaniment", "instrum"]
        vocal_keywords = ["vocal", "voice", "voc"]

        for url in urls:
            url_lower = url.lower()
            if any(kw in url_lower for kw in instrumental_keywords):
                self._log_msg(f"Selected instrumental URL: {url}")
                return url

        for url in urls:
            url_lower = url.lower()
            if not any(kw in url_lower for kw in vocal_keywords):
                self._log_msg(f"Selected non-vocal URL: {url}")
                return url

        self._log_msg(f"WARNING: No instrumental URL found, using first URL: {urls[0]}")
        return urls[0]

    def _build_mirror_url(self, original_url: str) -> str:
        parsed = urlparse(original_url)
        mirror_parsed = parsed._replace(netloc="mirror.mvsep.com")
        return urlunparse(mirror_parsed)

    def _try_download(self, url: str, output_path: str, job: MVSepJob,
                      progress_callback: Optional[Callable[[float], None]] = None) -> str:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

        resp = self._retry_request(
            requests.get, url,
            stream=True, timeout=TIMEOUT_DOWNLOAD,
        )

        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0

        with open(output_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if self.is_cancelled:
                    f.close()
                    if os.path.exists(output_path):
                        os.remove(output_path)
                    job.status = "cancelled"
                    raise MVSepAPIError("Download cancelled")

                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0 and progress_callback:
                        progress_callback((downloaded / total_size) * 100)

        self._log_msg(f"Download complete: {output_path}")
        return output_path

    def download_result(self, job: MVSepJob, output_path: str, progress_callback: Optional[Callable[[float], None]] = None) -> str:
        if not job.download_urls:
            raise MVSepAPIError("No download URLs available")

        self._log_msg(f"Downloading result to {os.path.basename(output_path)}...")
        job.status = "downloading"

        download_url = self._select_instrumental_url(job.download_urls)
        if not download_url:
            raise MVSepAPIError("No instrumental URL found in MVSep response")

        mirror_url = self._build_mirror_url(download_url)
        if mirror_url == download_url:
            mirror_url = ""

        try:
            return self._try_download(download_url, output_path, job, progress_callback)
        except Exception as e:
            if not mirror_url:
                raise
            self._log_msg(f"Primary download failed ({e}), trying mirror...")
            if os.path.exists(output_path):
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            try:
                return self._try_download(mirror_url, output_path, job, progress_callback)
            except Exception as mirror_e:
                self._log_msg(f"Mirror download also failed ({mirror_e})")
                raise MVSepAPIError(f"Download failed from both primary and mirror: {e}") from e

    def process_file(self, file_path: str, model: MVSepModel, output_path: str,
                     upload_progress_cb: Optional[Callable[[float], None]] = None,
                     process_progress_cb: Optional[Callable[[float], None]] = None,
                     download_progress_cb: Optional[Callable[[float], None]] = None) -> str:
        job = self.upload_file(file_path, model)
        if upload_progress_cb:
            upload_progress_cb(100.0)

        job = self.wait_for_completion(job, progress_callback=process_progress_cb)

        if job.is_failed:
            raise MVSepAPIError(f"Processing failed: {job.error}")

        return self.download_result(job, output_path, progress_callback=download_progress_cb)
