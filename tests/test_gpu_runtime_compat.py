"""Unit tests for GPU compute capability detection and legacy runtime compatibility (Maxwell CC 5.x / Kepler)."""
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from backend import runtime_setup
from utils.model_utils import effective_use_amp
from ml_collections import ConfigDict


def test_pick_torch_line_architectures():
    """Verify that compute capability correctly routes to the right PyTorch build."""
    # Maxwell GPUs (GTX 900 series, GTX 750 / 750 Ti)
    assert runtime_setup.pick_torch_line([5.2]) == "cu118"
    assert runtime_setup.pick_torch_line([5.0]) == "cu118"

    # Kepler GPUs (GTX 600/700 series, Titan) — no PyTorch 2.x CUDA support
    assert runtime_setup.pick_torch_line([3.5]) == "cpu"
    assert runtime_setup.pick_torch_line([3.0]) == "cpu"

    # Pascal through Ada / Hopper (GTX 10-series to RTX 40-series)
    assert runtime_setup.pick_torch_line([6.0]) == "cu121"
    assert runtime_setup.pick_torch_line([6.1]) == "cu121"
    assert runtime_setup.pick_torch_line([7.5]) == "cu121"
    assert runtime_setup.pick_torch_line([8.0]) == "cu121"
    assert runtime_setup.pick_torch_line([8.6]) == "cu121"
    assert runtime_setup.pick_torch_line([8.9]) == "cu121"
    assert runtime_setup.pick_torch_line([9.0]) == "cu121"

    # Blackwell (RTX 50-series)
    assert runtime_setup.pick_torch_line([10.0]) == "cu128"
    assert runtime_setup.pick_torch_line([12.0]) == "cu128"

    # No GPU / empty
    assert runtime_setup.pick_torch_line([]) == "cpu"


def test_torch_specs_has_cu118():
    """Verify that cu118 is present in _TORCH_SPECS."""
    assert "cu118" in runtime_setup._TORCH_SPECS
    assert runtime_setup._TORCH_SPECS["cu118"] == [
        "torch==2.4.1", "torchvision==0.19.1", "torchaudio==2.4.1"
    ]


def test_effective_use_amp_maxwell_safety():
    """Verify that effective_use_amp disables AMP on CC < 6.0 (Maxwell)."""
    cfg = ConfigDict({"training": {"use_amp": True}})

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.cuda.get_device_capability", return_value=(5, 2)):
        # On Maxwell (CC 5.2), AMP must be turned off for float32 safety
        assert effective_use_amp(cfg) is False

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.current_device", return_value=0), \
         patch("torch.cuda.get_device_capability", return_value=(8, 6)):
        # On Ampere (CC 8.6), AMP should be preserved
        assert effective_use_amp(cfg) is True


def test_attend_maxwell_safety():
    """Verify that BS-Roformer Attend disables mem_efficient attention on CC < 6.0."""
    from models.bs_roformer.attend import Attend

    mock_props = MagicMock()
    mock_props.major = 5
    mock_props.minor = 2

    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.get_device_properties", return_value=mock_props):
        attn = Attend(flash=True)
        assert attn.cuda_config is not None
        assert attn.cuda_config.enable_flash is False
        assert attn.cuda_config.enable_math is True
        # Memory-efficient (Cutlass) must be False on Maxwell to prevent driver crash!
        assert attn.cuda_config.enable_mem_efficient is False


def test_runtime_dialog_combobox_options():
    """Verify that RuntimeSetupDialog renders the expected build choices."""
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])

    from ui.widgets.runtime_dialog import RuntimeSetupDialog

    with patch("backend.runtime_setup.detect_gpus", return_value=[("GeForce GTX 960", 5.2)]):
        dlg = RuntimeSetupDialog(None, top_up=False)
        assert hasattr(dlg, "_line_combo")
        items = [dlg._line_combo.itemData(i) for i in range(dlg._line_combo.count())]
        assert "cu118" in items
        assert "cu121" in items
        assert "cu128" in items
        assert "cpu" in items
        # Because detect_gpus returned CC 5.2, the selected item should be cu118
        selected_line = dlg._line_combo.itemData(dlg._line_combo.currentIndex())
        assert selected_line == "cu118"
