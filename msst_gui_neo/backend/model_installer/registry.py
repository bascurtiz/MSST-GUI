"""backend/model_installer/registry.py
Defines the required local models for Iterative Ensemble.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class IterativeModel:
    id: str
    name: str
    arch: str
    stem_type: str
    ckpt_url: str
    yaml_url: str
    ckpt_filename: str
    yaml_filename: str
    subfolder: str = "bs_roformer"
    backend_module: str = ""


REQUIRED_MODELS = [
    IterativeModel(
        id="melband_v1e",
        name="MelBand RoFormer v1e",
        arch="Melband Roformer Architecture",
        stem_type="instrumental",
        ckpt_url="https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/inst_v1e.ckpt?download=true",
        yaml_url="https://huggingface.co/pcunwa/Mel-Band-Roformer-Inst/resolve/main/config_melbandroformer_inst.yaml?download=true",
        ckpt_filename="inst_v1e.ckpt",
        yaml_filename="config_melbandroformer_inst.yaml",
        subfolder="bs_roformer",
    ),
    IterativeModel(
        id="bs_resurrect",
        name="BS RoFormer Resurrect",
        arch="BS Roformer Architecture",
        stem_type="instrumental",
        ckpt_url="https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst.ckpt?download=true",
        yaml_url="https://huggingface.co/pcunwa/BS-Roformer-Resurrection/resolve/main/BS-Roformer-Resurrection-Inst-Config.yaml?download=true",
        ckpt_filename="BS-Roformer-Resurrection-Inst.ckpt",
        yaml_filename="BS-Roformer-Resurrection-Inst-Config.yaml",
        subfolder="bs_roformer",
    ),
    IterativeModel(
        id="bs_largev1",
        name="bs_largev1",
        arch="BS Roformer Architecture",
        stem_type="vocals",
        ckpt_url="https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_large1_unwa.ckpt?download=true",
        yaml_url="https://huggingface.co/noblebarkrr/mvsepless_resources/resolve/main/bs_roformer/bs_vocals_large1_unwa_config.yaml?download=true",
        ckpt_filename="BS-Roformer_LargeV1.ckpt",
        yaml_filename="config_bsrofoL.yaml",
        subfolder="bs_roformer",
    ),
]
