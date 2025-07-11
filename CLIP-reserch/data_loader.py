from utils.config import Config
from typing import Tuple
from torch.utils.data import DataLoader
from dataset import MistDataSet

def prepare_dataloaders(config: Config, resolution: int) -> Tuple[DataLoader, DataLoader]:
    train_set = MistDataSet(config=config, train=True, model_resolution=resolution)
    test_set = MistDataSet(config=config, train=False, model_resolution=resolution)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size)
    return train_loader, test_loader