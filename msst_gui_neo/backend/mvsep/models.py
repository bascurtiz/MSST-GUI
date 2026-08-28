"""backend/mvsep/models.py"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class MVSepModel(str, Enum):
    BS_ROFORMER_2025_07 = "bs_roformer_2025_07"
    SCNET_XL_IHF_BECRUILY = "scnet_xl_ihf_becruily"

    @property
    def sep_type(self) -> int:
        mapping = {
            "bs_roformer_2025_07": 40,
            "scnet_xl_ihf_becruily": 46,
        }
        return mapping.get(self.value, 40)

    @property
    def add_opt1(self) -> str:
        mapping = {
            "bs_roformer_2025_07": "81",
            "scnet_xl_ihf_becruily": "6",
        }
        return mapping.get(self.value, "")


@dataclass
class MVSepJob:
    task_hash: str
    model: MVSepModel
    status: str = "pending"
    upload_progress: float = 0.0
    processing_progress: float = 0.0
    download_urls: list = field(default_factory=list)
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None

    @property
    def is_complete(self) -> bool:
        return self.status == "done"

    @property
    def is_failed(self) -> bool:
        return self.status in ("failed", "error")

    @property
    def is_processing(self) -> bool:
        return self.status in ("pending", "uploading", "processing")
