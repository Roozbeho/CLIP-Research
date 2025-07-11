

from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class Config:
    device: torch.device
    model_name: str
    batch_size: int
    linear_probe_c: float
    linear_probe_max_iter: int
    dataset_root: str

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> 'Config':
        return cls(**conf)