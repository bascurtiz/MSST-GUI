"""
tests/test_model_registration_detection.py
-----------------------------------------
Unit tests verifying automatic detection of Architecture and Model Type
from config YAML/JSON files and URLs during model registration.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.yaml_analyzer import (
    classify_yaml_config,
    classify_model_arch,
    classify_model_type,
    ARCH_TYPES,
    MODEL_TYPES,
)


class TestModelRegistrationDetection(unittest.TestCase):
    def test_apollo_detection(self):
        apollo_cfg = os.path.join("configs", "config_apollo.yaml")
        if os.path.exists(apollo_cfg):
            res = classify_yaml_config(apollo_cfg, filename="config_apollo_uni.yaml")
            self.assertEqual(res["arch"], "Apollo Architecture")
            self.assertEqual(res["type"], "super resolution")

    def test_bs_roformer_instrumental_detection(self):
        inst_cfg = os.path.join("configs", "BS-Roformer-Resurrection-Inst-Config.yaml")
        if os.path.exists(inst_cfg):
            res = classify_yaml_config(inst_cfg, filename="BS-Roformer-Resurrection-Inst-Config.yaml")
            self.assertEqual(res["arch"], "BS Roformer Architecture")
            self.assertEqual(res["type"], "instrumental")

    def test_bs_roformer_vocals_detection(self):
        voc_cfg = os.path.join("configs", "config_vocals_bs_roformer.yaml")
        if os.path.exists(voc_cfg):
            res = classify_yaml_config(voc_cfg, filename="config_vocals_bs_roformer.yaml")
            self.assertEqual(res["arch"], "BS Roformer Architecture")
            self.assertEqual(res["type"], "vocals")

    def test_melband_roformer_detection(self):
        mbr_cfg = os.path.join("configs", "config_melbandroformer_inst.yaml")
        if os.path.exists(mbr_cfg):
            res = classify_yaml_config(mbr_cfg, filename="config_melbandroformer_inst.yaml")
            self.assertEqual(res["arch"], "Melband Roformer Architecture")
            self.assertEqual(res["type"], "instrumental")

    def test_bsmamba2_detection(self):
        mamba_cfg = os.path.join("configs", "config_bs_mamba2_vocals.yaml")
        if os.path.exists(mamba_cfg):
            res = classify_yaml_config(mamba_cfg, filename="config_bs_mamba2_vocals.yaml")
            self.assertEqual(res["arch"], "BSMamba2 Architecture")
            self.assertEqual(res["type"], "vocals")

    def test_demucs_detection(self):
        demucs_cfg = os.path.join("configs", "config_htdemucs_6stems.yaml")
        if os.path.exists(demucs_cfg):
            res = classify_yaml_config(demucs_cfg, filename="config_htdemucs_6stems.yaml")
            self.assertEqual(res["arch"], "Demucs Architecture")
            self.assertEqual(res["type"], "multi stems")

    def test_mdx23c_detection(self):
        mdx_cfg = os.path.join("configs", "mdx23c_instvoc_zfturbo_config.yaml")
        if os.path.exists(mdx_cfg):
            res = classify_yaml_config(mdx_cfg, filename="mdx23c_instvoc_zfturbo_config.yaml")
            self.assertEqual(res["arch"], "MDX23c Architecture")
            self.assertEqual(res["type"], "dual target (instrumental & vocals)")

    def test_url_string_detection(self):
        res_apollo = classify_yaml_config(None, filename="https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/models/apollo/apollo_model_uni.ckpt")
        self.assertEqual(res_apollo["arch"], "Apollo Architecture")
        self.assertEqual(res_apollo["type"], "super resolution")

        res_mdx = classify_yaml_config(None, filename="https://huggingface.co/user/repo/resolve/main/mdx23c_vocals.yaml")
        self.assertEqual(res_mdx["arch"], "MDX23c Architecture")
        self.assertEqual(res_mdx["type"], "vocals")

        res_denoise = classify_yaml_config(None, filename="https://huggingface.co/user/repo/resolve/main/bs_roformer_denoise.yaml")
        self.assertEqual(res_denoise["arch"], "BS Roformer Architecture")
        self.assertEqual(res_denoise["type"], "denoise")

    def test_settings_page_auto_detection_and_pairing(self):
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
        from PySide6.QtWidgets import QApplication
        _app = QApplication.instance() or QApplication(sys.argv)
        from ui.pages.settings_page import SettingsPage

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = os.path.join(tmpdir, "apollo_model_uni.ckpt")
            yaml_path = os.path.join(tmpdir, "config_apollo_uni.yaml")
            with open(ckpt_path, "wb") as f:
                f.write(b"dummy ckpt")
            apollo_cfg = os.path.join("configs", "config_apollo.yaml")
            if os.path.exists(apollo_cfg):
                with open(apollo_cfg, "rb") as f:
                    content = f.read()
            else:
                content = b"model:\n  sr: 44100\n  win: 20\n  feature_dim: 256\n  layer: 6\ntraining:\n  instruments: ['restored', 'addition']\n  target_instrument: 'restored'\n"
            with open(yaml_path, "wb") as f:
                f.write(content)

            page = SettingsPage()
            page._set_mode("local")

            # 1. Select YAML first
            page._yaml.set_value(yaml_path)
            self.assertEqual(page._ckpt.value(), os.path.normpath(ckpt_path))
            self.assertEqual(page._arch_combo.currentText(), "Apollo Architecture")
            self.assertEqual(page._type_combo.currentText(), "super resolution")

            # 2. Select CKPT first
            page2 = SettingsPage()
            page2._set_mode("local")
            page2._ckpt.set_value(ckpt_path)
            self.assertEqual(page2._yaml.value(), os.path.normpath(yaml_path))
            self.assertEqual(page2._arch_combo.currentText(), "Apollo Architecture")
            self.assertEqual(page2._type_combo.currentText(), "super resolution")

            # 3. URL mode detection
            page3 = SettingsPage()
            page3._set_mode("url")
            page3._yaml_url.set_value("https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/config_apollo_uni.yaml")
            self.assertEqual(page3._arch_combo.currentText(), "Apollo Architecture")
            self.assertEqual(page3._type_combo.currentText(), "super resolution")


if __name__ == "__main__":
    unittest.main()
