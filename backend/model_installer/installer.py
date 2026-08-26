"""backend/model_installer/installer.py
Downloads missing models using plain Python threads + requests.
No QThread, no QEventLoop, no blocking.
"""
import os
import shutil
import tempfile
import threading
from PySide6.QtCore import QObject, Signal

from backend.paths import APP_DIR
from backend.model_installer.registry import IterativeModel
from backend.download_utils import parallel_download


class ModelInstaller(QObject):
    progress = Signal(str, int, int)
    status = Signal(str)
    model_finished = Signal(str, dict)
    model_error = Signal(str, str)
    all_finished = Signal()

    def __init__(self):
        super().__init__()
        self._cancel_requested = False
        self._lock = threading.Lock()

    def cancel(self):
        with self._lock:
            self._cancel_requested = True

    @property
    def is_cancelled(self):
        with self._lock:
            return self._cancel_requested

    def _get_dest_paths(self, model):
        sub = os.path.join(APP_DIR, "models", model.subfolder) if model.subfolder else os.path.join(APP_DIR, "models")
        os.makedirs(sub, exist_ok=True)
        ckpt_dest = os.path.join(sub, model.ckpt_filename)
        yaml_dest = os.path.join(APP_DIR, "configs", model.yaml_filename)
        os.makedirs(os.path.dirname(yaml_dest), exist_ok=True)
        return ckpt_dest, yaml_dest

    def _download_file(self, url, dest_path, model_id):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        ok, msg = parallel_download(
            url, dest_path,
            progress_callback=lambda n, c, t: self.progress.emit(model_id, c, t),
            should_cancel=lambda: self.is_cancelled,
            timeout=(300, 60),
        )
        if not ok:
            self.status.emit("Download failed: " + msg)
            return False
        return True

    def _do_install(self, models_to_install):
        for model in models_to_install:
            with self._lock:
                if self._cancel_requested:
                    break
            self._install_one(model)
        with self._lock:
            if not self._cancel_requested:
                self.all_finished.emit()

    def _install_one(self, model):
        with self._lock:
            if self._cancel_requested:
                return

        ckpt_dest, yaml_dest = self._get_dest_paths(model)
        need_ckpt = not os.path.isfile(ckpt_dest)
        need_yaml = not os.path.isfile(yaml_dest)

        if not need_ckpt and not need_yaml:
            self.status.emit(model.name + ": already installed")
            self.model_finished.emit(model.id, {
                "name": model.ckpt_filename,
                "ckpt": ckpt_dest,
                "yaml": yaml_dest,
                "arch": model.arch,
                "type": model.stem_type,
                "backend_module": getattr(model, "backend_module", ""),
                "custom_backend_enabled": False,
            })
            return

        temp_dir = tempfile.mkdtemp(prefix="msst_iterative_install_")

        try:
            if need_ckpt:
                with self._lock:
                    if self._cancel_requested:
                        return
                self.status.emit(model.name + ": downloading checkpoint...")
                ckpt_temp = os.path.join(temp_dir, model.ckpt_filename)

                success = self._download_file(model.ckpt_url, ckpt_temp, model.id)
                with self._lock:
                    cancelled = self._cancel_requested
                if not success or cancelled:
                    self.model_error.emit(model.id, "Download cancelled or failed")
                    return

                self.status.emit(model.name + ": installing checkpoint...")
                shutil.copy2(ckpt_temp, ckpt_dest)

            if need_yaml:
                with self._lock:
                    if self._cancel_requested:
                        return
                self.status.emit(model.name + ": downloading config...")
                yaml_temp = os.path.join(temp_dir, model.yaml_filename)

                success = self._download_file(model.yaml_url, yaml_temp, model.id)
                with self._lock:
                    cancelled = self._cancel_requested
                if not success or cancelled:
                    self.model_error.emit(model.id, "Download cancelled or failed")
                    return

                self.status.emit(model.name + ": installing config...")
                shutil.copy2(yaml_temp, yaml_dest)

            self.status.emit(model.name + ": installation complete")
            self.model_finished.emit(model.id, {
                "name": model.ckpt_filename,
                "ckpt": ckpt_dest,
                "yaml": yaml_dest,
                "arch": model.arch,
                "type": model.stem_type,
                "backend_module": getattr(model, "backend_module", ""),
                "custom_backend_enabled": False,
            })

        except Exception as e:
            self.model_error.emit(model.id, str(e))
        finally:
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass

    def install_all(self, models_to_install):
        t = threading.Thread(target=self._do_install, args=(models_to_install,), daemon=True)
        t.start()
