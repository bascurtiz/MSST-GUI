"""Regression test for the job-start runtime gate.

A refactor once left a stale helper call inside ensure_runtime(), which only
exploded when the user clicked RUN INFERENCE — compile checks don't catch
runtime NameErrors, so we exercise it here directly.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402


def test_runtime_gate_works_in_dev():
    app = QApplication.instance() or QApplication([])
    from ui.widgets import runtime_dialog as rd

    # calling these must not raise NameError/AttributeError
    assert rd.runtime_usable() is True          # dev checkout is always usable
    assert rd.ensure_runtime(None) is True      # gate passes without a dialog
