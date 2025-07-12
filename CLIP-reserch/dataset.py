from typing import Tuple

import torch
from utils.config import Config
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

class MistDataSet(Dataset):
    """
    Custom dataset class for MNIST dataset.

    Attributes:
    ----------
    config : Config
        Configuration object containing dataset settings.
    base_data : torchvision.datasets.MNIST
        Base MNIST dataset object.
    model_resolution : int
        Resolution of the model (used for image transformations, same as transforms returned by clip.load()).
    transform : transforms.Compose
        Composition of image transformations.

    Methods:
    -------
    __init__(config: Config, train: bool = True, download: bool = True, model_resolution: int = None)
        Initializes the dataset object.
    __len__() -> int
        Returns the number of samples in the dataset.
    __getitem__(idx: int) -> Tuple[torch.Tensor, int]
        Returns a sample from the dataset at the specified index.
    _get_transform(n_px: int) -> transforms.Compose
        Returns a composition of image transformations.
    """
    def __init__(
            self,
            config: Config,
            train: bool = True,
            download: bool = True,
            model_resolution = None
        ):
        """
        Initializes the dataset object.

        Parameters:
        ----------
        config : Config
            Configuration object containing dataset settings.
        train : bool, optional
            Flag to use training or testing set (default: True).
        download : bool, optional
            Flag to download the dataset if not already present (default: True).
        model_resolution : int, optional
            Resolution of the model (used for image transformations) (default: None).
        """
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
        """
        Returns a composition of image transformations.
        it's same as tranform function that returned from clip.load(),
        transforms images in data procces part for performance concerns

        Parameters:
        ----------
        n_px : int
            Resolution of the model.
        """
        return transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ]) 