"""Unit tests for checkpoint settings defaults, reset, and chunk size handling."""
import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from ui.widgets.ckpt_settings_dialog import get_model_yaml_defaults, CkptSettingsDialog
from backend import settings as settings_store
from PySide6.QtWidgets import QApplication


def test_get_model_yaml_defaults_apollo():
    """Apollo model YAML has no inference section; verify true defaults are extracted."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config_apollo_uni.yaml")
    defaults = get_model_yaml_defaults(yaml_path, "Apollo Architecture")
    assert defaults["overlap"] == 2
    assert defaults["batch_size"] == 1
    assert defaults["chunk_size"] == 132300


def test_get_model_yaml_defaults_roformer():
    """BS-Roformer YAML defaults should match config."""
    yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "viperx", "model_bs_roformer_ep_317_sdr_12.9755.yaml")
    if os.path.isfile(yaml_path):
        defaults = get_model_yaml_defaults(yaml_path, "BS Roformer Architecture")
        assert defaults["overlap"] == 2
        assert defaults["batch_size"] == 4
        assert defaults["chunk_size"] == 352800


def test_ckpt_settings_dialog_initialization_and_reset():
    """Verify dialog sliders start at model defaults and reset correctly."""
    app = QApplication.instance() or QApplication([])

    yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config_apollo_uni.yaml")
    dlg = CkptSettingsDialog("apollo_model_uni.ckpt", yaml_path, "Apollo Architecture")

    # Initial slider values must match Apollo defaults (not 8 or 44100!)
    assert dlg._overlap_slider.get_value() == 2
    assert dlg._batch_slider.get_value() == 1
    assert dlg._chunk_slider.get_value() == 132300

    # Simulate user changing values
    dlg._overlap_slider.set_value(8)
    dlg._batch_slider.set_value(4)
    dlg._chunk_slider.set_value(44100)
    assert dlg._overlap_slider.get_value() == 8

    # Reset to defaults
    dlg._reset_to_defaults()
    assert dlg._overlap_slider.get_value() == 2
    assert dlg._batch_slider.get_value() == 1
    assert dlg._chunk_slider.get_value() == 132300


def test_ckpt_settings_save_and_delete():
    """Verify that saving default settings emits empty dict and removes the override."""
    app = QApplication.instance() or QApplication([])

    yaml_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config_apollo_uni.yaml")
    dlg = CkptSettingsDialog("test_apollo.ckpt", yaml_path, "Apollo Architecture")

    saved_payload = []
    dlg.settings_saved.connect(lambda name, data: saved_payload.append((name, data)))

    # Save when values match defaults -> should emit empty dict {}
    dlg._save()
    assert len(saved_payload) == 1
    assert saved_payload[0] == ("test_apollo.ckpt", {})

    # Test settings_store.save_ckpt_settings removes entry when empty dict passed
    settings_store.save_ckpt_settings("test_apollo.ckpt", {"overlap": 4})
    assert "test_apollo.ckpt" in settings_store.load_ckpt_settings()

    settings_store.save_ckpt_settings("test_apollo.ckpt", {})
    assert "test_apollo.ckpt" not in settings_store.load_ckpt_settings()


def test_get_ckpt_settings_variations():
    """Verify that get_ckpt_settings matches by name, basename, stem, and case-insensitively."""
    settings_store.save_ckpt_settings("mbr_karaoke_becruily.ckpt", {"overlap": 1, "chunk_size": 352800})
    try:
        # Match by exact name with .ckpt
        assert settings_store.get_ckpt_settings("mbr_karaoke_becruily.ckpt")["overlap"] == 1
        # Match by stem without .ckpt
        assert settings_store.get_ckpt_settings("mbr_karaoke_becruily")["overlap"] == 1
        # Match by full path
        assert settings_store.get_ckpt_settings("Random Display Name", r"E:\models\melband_roformer\mbr_karaoke_becruily.ckpt")["overlap"] == 1
        # Case insensitive match
        assert settings_store.get_ckpt_settings("MBR_KARAOKE_BECRUILY.CKPT")["overlap"] == 1
        # Non-existent
        assert settings_store.get_ckpt_settings("non_existent_model") == {}
    finally:
        settings_store.save_ckpt_settings("mbr_karaoke_becruily.ckpt", {})


def test_inference_page_global_overlap():
    """Verify that InferencePage has global Overlap control in Optional fold and saves/loads it."""
    from ui.pages.inference_page import InferencePage
    app = QApplication.instance() or QApplication([])

    page = InferencePage()
    assert hasattr(page, "_overlap_combo")
    items = [page._overlap_combo.itemText(i) for i in range(page._overlap_combo.count())]
    assert items == ["Default", "1", "2", "3", "4"]

    # Test save_settings includes overlap
    page._overlap_combo.setCurrentText("1")
    saved = page.save_settings()
    assert saved.get("overlap") == "1"

    # Test load_settings restores overlap
    page.load_settings({"overlap": "2"})
    assert page._overlap_combo.currentText() == "2"

    page.load_settings({"overlap": "Default"})
    assert page._overlap_combo.currentText() == "Default"
    page.deleteLater()


