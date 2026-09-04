"""Regression test: ensemble runners pre-flight the engine branch check.

Both ensemble pipelines spawn inference.py per model. Before this change a
model whose stored model_type had no engine branch only failed inside each
subprocess (raw 'Unknown model type' traceback), after the job already
started and other models ran. Now each runner aborts the WHOLE job up front
with a clear console error, before any subprocess or side effect.

Drives the real AutoEnsembleRunner and IterativeEnsembleRunner synchronously
(run() in-thread) offscreen — no torch, no network, no subprocesses are ever
spawned because the abort happens first.
"""
import os
import shutil
import sys
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402

from backend.auto_ensemble_runner import AutoEnsembleRunner  # noqa: E402
from backend.iterative_ensemble.runner import IterativeEnsembleRunner  # noqa: E402

FAILURES = []


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    if not cond:
        FAILURES.append(name)


def model(name, mtype, enabled=True):
    return {"name": name, "ckpt": "C:/nope/m.ckpt", "yaml": "C:/nope/m.yaml",
            "arch": "MDX23c Architecture", "model_type": mtype,
            "type": "vocals", "enabled": enabled}


def main():
    app = QApplication([])

    # ── Auto ensemble ──────────────────────────────────────────────────────
    # Control: all-supported models pass the pre-flight (the run then ends in
    # its ordinary empty-input abort, NOT a model-type abort).
    c_logs, c_fin = [], []
    c = AutoEnsembleRunner([model("G1", "mdx23c"), model("G2", "bs_roformer")],
                           [], "instrumental", "avg", "C:/nope/out")
    c.log_line.connect(c_logs.append)
    c.finished.connect(lambda ok, msg, path: c_fin.append((ok, msg)))
    c.run()
    check("auto control: preflight passes (no type error)",
          not any("cannot be run" in l for l in c_logs))
    check("auto control: proceeds to normal empty-input abort",
          bool(c_fin) and c_fin[0][1] == "No audio files found in input.")

    # Abort: one unsupported model aborts the whole job pre-spawn.
    logs, fin = [], []
    r = AutoEnsembleRunner([model("Ok", "mdx23c"),
                            model("Future Zoo", "hyperace_v9000")],
                           "C:/nope/song.mp3", "instrumental", "avg",
                           "C:/nope/out")
    r.log_line.connect(logs.append)
    r.finished.connect(lambda ok, msg, path: fin.append((ok, msg)))
    r.run()
    joined = "\n".join(logs)
    check("auto: aborted with failure", bool(fin) and not fin[0][0])
    check("auto: clear error names the model + type",
          "model 'Future Zoo' cannot be run: model type 'hyperace_v9000'"
          in joined)
    check("auto: supported types listed", "Supported model types:" in joined)
    check("auto: abort message counts offenders",
          fin and "Aborted: 1 model(s) cannot run" in fin[0][1])
    check("auto: aborted before temp dir creation",
          getattr(r, "_temp_dir", None) is None)

    # ── Iterative ensemble ─────────────────────────────────────────────────
    # Control: a supported local model proceeds past the pre-flight.
    scratch = tempfile.mkdtemp(prefix="msst_preflight_")
    good_cfg = {"input_files": [], "output_dir": os.path.join(scratch, "good"),
                "iterations": 2, "models_local": {"m": model("Good", "mdx23c")},
                "models_api": {}, "finisher_variants": ["mvsep_only"]}
    ic_logs, ic_fin = [], []
    ic = IterativeEnsembleRunner(good_cfg)
    ic.log_line.connect(ic_logs.append)
    ic.finished.connect(lambda ok, msg, path: ic_fin.append((ok, msg)))
    ic.run()
    check("iter control: preflight passes (no type error)",
          not any("cannot be run" in l for l in ic_logs))
    check("iter control: pipeline completes normally",
          bool(ic_fin) and ic_fin[0][0] and "complete" in ic_fin[0][1])

    # Abort: an enabled unsupported model aborts the whole pipeline; disabled
    # models are ignored, exactly like the per-pass local-model filter.
    bad_cfg = {"input_files": [], "output_dir": os.path.join(scratch, "bad"),
               "iterations": 2,
               "models_local": {
                   "ok": model("Ok", "mdx23c"),
                   "bad": model("Future Zoo", "hyperace_v9000"),
                   "off": model("Disabled", "hyperace_v9000", enabled=False),
               },
               "models_api": {}, "finisher_variants": ["mvsep_only"]}
    ilogs, ifin = [], []
    ir = IterativeEnsembleRunner(bad_cfg)
    ir.log_line.connect(ilogs.append)
    ir.finished.connect(lambda ok, msg, path: ifin.append((ok, msg)))
    ir.run()
    ijoined = "\n".join(ilogs)
    check("iter: aborted with failure", bool(ifin) and not ifin[0][0])
    check("iter: only the enabled unsupported model flagged",
          ijoined.count("model 'Future Zoo' cannot be run:") == 1
          and "Disabled" not in ijoined)
    check("iter: abort message counts offenders",
          ifin and "Aborted: 1 model(s) cannot run" in ifin[0][1])
    check("iter: output dir never created (aborted pre-spawn)",
          not os.path.exists(bad_cfg["output_dir"]))
    shutil.rmtree(scratch, ignore_errors=True)

    app.processEvents()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURES:")
        for f in FAILURES:
            print(" -", f)
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
