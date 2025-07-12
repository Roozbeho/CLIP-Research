

from dataclasses import dataclass
from typing import Dict, Any

import torch


@dataclass
class Config:
    """
    Configuration class for model settings.

    Attributes:
    ----------
    device : torch.device
        Device to run the model on (GPU or CPU).
    model_name : str
        Name of the model to use.
    batch_size : int
        Batch size for model training.
    linear_probe_c : float
        Hyperparameter for linear probe.
    linear_probe_max_iter : int
        Maximum number of iterations for linear probe.
    dataset_root : str
        Root directory of the dataset.
    save_visualization : bool
        Flag to save visualizations.

    Methods:
    -------
    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> 'Config':
        Creates a Config instance from a dictionary.

    Returns:
    -------
    Config
        A Config instance with the specified settings.
    """
    device: torch.device
    model_name: str
    batch_size: int
    linear_probe_c: float
    linear_probe_max_iter: int
    dataset_root: str
    save_visualization: bool

    @classmethod
    def from_dict(cls, conf: Dict[str, Any]) -> 'Config':
        return cls(**conf)