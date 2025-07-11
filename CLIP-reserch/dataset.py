from typing import Tuple

import torch
from utils.config import Config
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class MistDataSet(Dataset):
    def __init__(
            self,
            config: Config,
            train: bool = True,
            download: bool = True,
            model_resolution = None
        ):
        self.config = config
        self.base_data = torchvision.datasets.MNIST(root=config.dataset_root, train=train, download=download)
        self.model_resolution = model_resolution
        self.transform = self._get_transform(model_resolution)
    
    def __len__(self) -> int:
        return len(self.base_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base_data[idx]

        if self.transform:
            img = self.transform(img)
        
        return img, label

    @staticmethod
    def _get_transform(n_px: int) -> transforms.Compose:
        return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) 