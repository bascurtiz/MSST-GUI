"""Regression test: BATCH MODEL MODE batch flow (inference page).

Arming multi-select ("Batch model mode") then ticking every model via
Select all and pressing Separate must process models in name order, continue
past per-model failures, spawn exactly one inference job per *supported*
model, and never spawn for models a pre-run guard blocks (missing files, no
engine branch). Exercises the real InferencePage (._toggle_multi_mode,
._on_select_all_toggled, ._run -> ._run_multi_selection) offscreen with a
stubbed settings store and a stubbed ProcessRunner that reports success — no
torch, no network, no subprocess.

Models under test (registry-entry shape):
  * a_future_zoo     — supported-type files present, engine type unknown to
                       this build  -> blocked by the type guard
  * m_mid_supported  — files present, supported type  -> runner spawned,
                       finished(0) -> TEST OK
  * z_missing_files  — ckpt/yaml gone from disk -> blocked by the file guard
"""
import os
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.theme import theme_manager  # noqa: E402
import ui.pages.inference_page as ip  # noqa: E402
import backend.settings as bs  # noqa: E402

FAILURES = []


class _FakeMsgBox:
    """Stand-in for ui.pages.inference_page.QMessageBox: records every
    warning / info instead of opening a modal dialog offscreen."""
    warns = []

    @staticmethod
    def warning(parent, title, text):
        _FakeMsgBox.warns.append(text)
        return None

    @staticmethod
    def information(parent, title, text):
        _FakeMsgBox.warns.append(text)
        return None


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


class _NamesFetchThread(ip._NamesFetchThread):
    """Never touch the network in the harness."""
    def start(self):
        pass


class _StubRunner(ip.ProcessRunner):
    """Stand-in for the real QProcess runner: records the launch command and
    reports an immediate success through the same signal contract."""
    spawned = []

    def __init__(self, cmd, cwd=None):
        _StubRunner.spawned.append((cmd, cwd))
        super().__init__(cmd, cwd=cwd)

    def start(self):
        from PySide6.QtCore import QTimer
        QTimer.singleShot(0, lambda: self.finished.emit(0))


def main():
    app = QApplication([])
    theme_manager.init_app(app)
    ip._NamesFetchThread = _NamesFetchThread
    ip.ProcessRunner = _StubRunner
    _StubRunner.spawned = []
    real_msgbox = ip.QMessageBox
    ip.QMessageBox = _FakeMsgBox
    _FakeMsgBox.warns = []

    tmp = tempfile.mkdtemp(prefix="msst_batch_test_")
    audio = os.path.join(tmp, "input.wav")
    with open(audio, "wb") as f:
        f.write(b"RIFF" + b"\0" * 256)  # never decoded: only the stub runs
    out = os.path.join(tmp, "out")
    os.makedirs(out)

    ckpt_a = os.path.join(tmp, "a.ckpt")
    yaml_a = os.path.join(tmp, "a.yaml")
    ckpt_m = os.path.join(tmp, "m.ckpt")
    yaml_m = os.path.join(tmp, "m.yaml")
    for p in (ckpt_a, yaml_a, ckpt_m, yaml_m):
        with open(p, "w", encoding="utf-8") as f:
            f.write("")

    models = [
        {"name": "z_missing_files", "ckpt": os.path.join(tmp, "gone.ckpt"),
         "yaml": os.path.join(tmp, "gone.yaml"),
         "arch": "MDX23c Architecture"},
        {"name": "m_mid_supported", "ckpt": ckpt_m, "yaml": yaml_m,
         "arch": "MDX23c Architecture", "model_type": "mdx23c",
         "type": "vocals"},
        {"name": "a_future_zoo", "ckpt": ckpt_a, "yaml": yaml_a,
         "arch": "MDX23c Architecture", "model_type": "hyperace_v9000",
         "type": "vocals"},
    ]

    real_load = bs.load

    def fake_load():
        d = dict(real_load())
        d["registered_models"] = list(models)
        return d

    def _register_all(page):
        """The app's main window feeds registered models into the page via
        on_model_registered; the harness must do the same so the library
        rows exist before Select all is exercised."""
        for m in models:
            page.on_model_registered(m)
        page._flush_library_visibility()

    bs.load = fake_load
    try:
        page = ip.InferencePage()
        _register_all(page)
        page._input_row.set_value([audio])
        page._output_row.set_value(out)
        placeholder = {"name": "placeholder"}
        page._selected_model = placeholder
        logs = []
        running = []
        page.log_output.connect(logs.append)
        page.process_running.connect(running.append)

        page._toggle_multi_mode()
        page._on_select_all_toggled(True)
        page._run()
        text = "\n".join(logs)

        # 1. Name-sorted processing order.
        idx = [text.index(f"[{i}/3] {n}") for i, n in
               ((1, "a_future_zoo"), (2, "m_mid_supported"),
                (3, "z_missing_files"))]
        check("models processed in sorted name order",
              idx[0] < idx[1] < idx[2])

        # 2. Type guard blocks the unsupported model before launch.
        check("type-guard ERROR logged",
              "model 'a_future_zoo' cannot be run: model type "
              "'hyperace_v9000' has no branch" in text)
        check("type-guard TEST FAIL reason",
              "TEST FAIL: model type 'hyperace_v9000' has no engine "
              "branch in this build" in text)

        # 3. Missing-files guard blocks the deleted model.
        check("missing-files ERROR logged",
              "ERROR: model 'z_missing_files' is missing its files on "
              "disk:" in text)
        check("missing-files TEST FAIL reason",
              "TEST FAIL: files missing on disk (see log above)" in text)

        # 4. Supported model spawns exactly once and reports OK.
        check("supported model spawned one runner",
              len(_StubRunner.spawned) == 1)
        cmd = _StubRunner.spawned[0][0] if _StubRunner.spawned else []
        check("launch cmd carries the supported ckpt",
              any("m.ckpt" in str(c) for c in cmd))
        check("launch cmd carries --config_path",
              "--config_path" in cmd)
        check("supported model TEST OK logged",
              "[2/3] m_mid_supported" in text and "  TEST OK" in text)

        # 5. Mixed summary names only the failures, in name order.
        check("summary header present", "TEST SUMMARY" in text)
        check("summary 1/3 with sorted failed names",
              "1/3 models passed \u2014 FAILED: a_future_zoo, z_missing_files"
              in text)

        # 6. State restored after the batch.
        check("prior selection restored", page._selected_model is placeholder)
        check("run button re-enabled", page.btn_run.isEnabled())
        check("test-all button re-enabled", page.btn_test_all.isEnabled())
        check("stop button disabled", not page.btn_stop.isEnabled())
        check("batch flag cleared", not page._batch_testing)
        check("process_running ended False",
              running and running[-1] is False)
        check("no warning shown for the valid batch",
              _FakeMsgBox.warns == [])

        # 7. The reported regression: batch must also run when the input and
        # output are set but NO library model is pre-selected — the batch
        # owns the selection per model. (Before the fix this path hit a
        # NameError on QMessageBox inside the slot and did nothing at all.)
        page2 = ip.InferencePage()
        _register_all(page2)
        page2._input_row.set_value([audio])
        page2._output_row.set_value(out)
        page2._selected_model = None
        logs2 = []
        page2.log_output.connect(logs2.append)
        page2._toggle_multi_mode()
        page2._on_select_all_toggled(True)
        page2._run()
        text2 = "\n".join(logs2)
        check("batch runs without a model pre-selected",
              "RUNNING SELECTED MODELS (3)" in text2
              and "TEST SUMMARY" in text2)
        check("no selection -> no warning dialog",
              _FakeMsgBox.warns == [])
        check("prior (None) selection restored",
              page2._selected_model is None)

        # 8. No input file -> a visible warning, clean state, no batch start.
        page3 = ip.InferencePage()
        _register_all(page3)
        page3._output_row.set_value(out)
        page3._selected_model = None
        logs3 = []
        page3.log_output.connect(logs3.append)
        page3._toggle_multi_mode()
        page3._on_select_all_toggled(True)
        page3._run()
        check("missing input shows a warning dialog",
              _FakeMsgBox.warns
              and "select at least one audio file" in _FakeMsgBox.warns[-1])
        check("missing input does not start the batch",
              not "RUNNING SELECTED MODELS" in "\n".join(logs3))
        check("test-all button still enabled after warning",
              page3.btn_test_all.isEnabled())

        # 9. Every batch model runs with ITS OWN stem context. A model whose
        # config pins a single target (bs_inst_large2_unwa: num_stems 1,
        # instruments [vocals, instrument], target_instrument "instrument")
        # must auto-derive the complement (--extract_instrumental) under the
        # default "all stems" selection; a plain multi-stem model must not.
        # Before the fix the batch reused the row state of the last manual
        # click (or an empty row), so the pinned model emitted only its
        # target stem.
        import yaml as _yaml
        ckpt_s = os.path.join(tmp, "bs_inst_large2_unwa.ckpt")
        yaml_s = os.path.join(tmp, "bs_inst_large2_unwa.yaml")
        ckpt_m4 = os.path.join(tmp, "m_four_stem.ckpt")
        yaml_m4 = os.path.join(tmp, "m_four_stem.yaml")
        for p in (ckpt_s, ckpt_m4):
            with open(p, "w", encoding="utf-8") as f:
                f.write("")
        with open(yaml_s, "w", encoding="utf-8") as f:
            _yaml.dump({"training": {"instruments": ["vocals", "instrument"],
                                     "target_instrument": "instrument"},
                        "model": {"num_stems": 1}}, f)
        with open(yaml_m4, "w", encoding="utf-8") as f:
            _yaml.dump({"training": {"instruments": ["vocals", "drums",
                                                      "bass", "other"]},
                        "model": {"num_stems": 4}}, f)
        models2 = [
            {"name": "bs_inst_large2_unwa.ckpt", "ckpt": ckpt_s,
             "yaml": yaml_s, "arch": "BS Roformer Architecture",
             "model_type": "bs_roformer", "type": "instrumental"},
            {"name": "m_four_stem.ckpt", "ckpt": ckpt_m4,
             "yaml": yaml_m4, "arch": "BS Roformer Architecture",
             "model_type": "bs_roformer", "type": "multi stems"},
        ]
        real_load2 = bs.load

        def fake_load2():
            d = dict(real_load2())
            d["registered_models"] = list(models2)
            return d

        _StubRunner.spawned = []
        warns_before_scenario9 = len(_FakeMsgBox.warns)
        bs.load = fake_load2
        try:
            page4 = ip.InferencePage()
            for m in models2:
                page4.on_model_registered(m)
            page4._flush_library_visibility()
            page4._input_row.set_value([audio])
            page4._output_row.set_value(out)
            page4._selected_model = None
            logs4 = []
            page4.log_output.connect(logs4.append)
            page4._toggle_multi_mode()
            page4._on_select_all_toggled(True)
            page4._run()
            text4 = "\n".join(logs4)
            check("per-stem batch runs both models",
                  "2/2 models passed" in text4)
            cmds = [c for c, _cw in _StubRunner.spawned]
            check("exactly two spawns for the stem-context batch",
                  len(cmds) == 2)
            single = next((c for c in cmds
                           if "bs_inst_large2_unwa.ckpt" in str(c)), None)
            multi = next((c for c in cmds
                          if "m_four_stem.ckpt" in str(c)), None)
            check("single-target model gets --extract_instrumental",
                  single is not None and "--extract_instrumental" in single)
            check("plain multi-stem model runs without the rest flag",
                  multi is not None and "--extract_instrumental" not in multi)
        finally:
            bs.load = real_load2
        check("no new warning shown for the stem-context batch",
              len(_FakeMsgBox.warns) == warns_before_scenario9)
    finally:
        bs.load = real_load
        ip.QMessageBox = real_msgbox


if __name__ == "__main__":
    main()
    print("FAILED: " + ", ".join(FAILURES) if FAILURES else "ALL CHECKS PASSED")
    sys.exit(1 if FAILURES else 0)
