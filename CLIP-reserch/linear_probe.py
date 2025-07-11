from typing import Tuple

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from utils.config import Config
import torch
from torch.utils.data import DataLoader
import numpy as np

class LinearProbe:
    def __init__(self, config:Config, clip_model: torch.nn.Module):
        self.config = config
        self.model = clip_model
    
    @torch.inference_mode()
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []

        with tqdm(total = len(dataloader), unit='Feature Extraction') as progress_bar:
            for images, labels in dataloader:
                images, labels = images.to(self.config.device), labels.to(self.config.device)

                features = self.model.encode_image(images)
                all_features.append(features)
                all_labels.append(labels)

                progress_bar.update(1)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
        
    def train_and_evaluate(self, train_loader: DataLoader, test_loader: DataLoader):
        train_features, train_labels = self.extract_features(train_loader)
        test_features, test_labels = self.extract_features(test_loader)

        clf = LogisticRegression(
            random_state=0,
            C=self.config.linear_probe_c,
            max_iter=self.config.linear_probe_max_iter,
            verbose=1
        )
        clf.fit(train_features, train_labels)

        preds = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, preds) * 100
        print(f"Linear Probe accuracy = {accuracy:.2f}")
