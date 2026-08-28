"""
backend/downloader.py
HuggingFace model downloader with progress tracking.
Supports resolve/main and blob/main URL formats.
"""
import os
import re
from urllib.parse import urlparse, unquote
from PySide6.QtCore import QObject, Signal

from backend.download_utils import parallel_download


class HuggingFaceDownloader(QObject):
    progress = Signal(str, int, int)   # file_name, bytes_downloaded, total_bytes
    status = Signal(str)               # status message
    finished = Signal(bool, str, dict) # success, message, file_info
    error = Signal(str)                # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def reset(self):
        self._cancel_requested = False

    @staticmethod
    def extract_filename(url):
        url = unquote(url)
        if "?" in url:
            url = url.split("?", 1)[0]
        if "/resolve/" in url:
            parts = url.split("/resolve/", 1)
            if len(parts) == 2:
                return os.path.basename(parts[1])
        elif "/blob/" in url:
            parts = url.split("/blob/", 1)
            if len(parts) == 2:
                return os.path.basename(parts[1])
        parsed = urlparse(url)
        path = parsed.path
        if path:
            return os.path.basename(path)
        return "downloaded_file"

    @staticmethod
    def is_hf_url(url):
        if not url:
            return False
        url = url.strip()
        return ("huggingface.co" in url or "hf.co" in url) and ("/resolve/" in url or "/blob/" in url)

    @staticmethod
    def convert_blob_to_resolve(url):
        if "/blob/" in url:
            return url.replace("/blob/", "/resolve/", 1)
        return url

    def download_file(self, url, dest_path):
        self._cancel_requested = False
        url = self.convert_blob_to_resolve(url.strip())
        filename = self.extract_filename(url)

        self.status.emit(f"Connecting to HuggingFace...")

        ok, msg = parallel_download(
            url, dest_path,
            progress_callback=lambda n, c, t: self.progress.emit(n, c, t),
            should_cancel=lambda: self._cancel_requested,
            timeout=(30, 30),
        )

        if not ok:
            if msg == "cancelled":
                self.status.emit("Download cancelled.")
                if os.path.exists(dest_path):
                    os.remove(dest_path)
            else:
                self.error.emit(msg)
            return False

        self.status.emit(f"Download complete: {filename}")
        return True

    def download_model(self, ckpt_url, yaml_url, ckpt_dest, yaml_dest):
        self._cancel_requested = False

        if not self.is_hf_url(ckpt_url):
            self.error.emit("Invalid checkpoint URL. Must be a HuggingFace resolve/main or blob/main link.")
            return False

        if not self.is_hf_url(yaml_url):
            self.error.emit("Invalid YAML URL. Must be a HuggingFace resolve/main or blob/main link.")
            return False

        self.status.emit("Starting download...")

        ckpt_filename = self.extract_filename(ckpt_url)
        yaml_filename = self.extract_filename(yaml_url)

        self.status.emit(f"Downloading checkpoint: {ckpt_filename}")
        if not self.download_file(ckpt_url, ckpt_dest):
            return False

        if self._cancel_requested:
            return False

        self.status.emit(f"Downloading config: {yaml_filename}")
        if not self.download_file(yaml_url, yaml_dest):
            try:
                os.remove(ckpt_dest)
            except OSError:
                pass
            return False

        if self._cancel_requested:
            try:
                os.remove(ckpt_dest)
                os.remove(yaml_dest)
            except OSError:
                pass
            return False

        file_info = {
            "ckpt_name": ckpt_filename,
            "yaml_name": yaml_filename,
            "ckpt_path": os.path.normpath(ckpt_dest),
            "yaml_path": os.path.normpath(yaml_dest),
        }

        self.status.emit("Download complete!")
        self.finished.emit(True, "Model downloaded successfully.", file_info)
        return True
