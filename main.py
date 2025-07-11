import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List, Dict, Any
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from dataclasses import dataclass

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

class ZeroShotClassification:
    def __init__(self, config: Config, clip_model: torch.nn.Module, classes: List[List[str]]):
        self.config = config
        self.model = clip_model
        self.classes = classes
        self.tokenized_classes: List[torch.Tensor] = [
            clip.tokenize(target_cls).to(self.config.device)
            for target_cls in self.classes
        ]
    
    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> List[int]:
        image_features = self.model.encode_image(images)
        # Image normalization
        image_features /= image_features.norm(dim=-1, keepdim=True)

        classes_probs_argmax: List[int] = []

        for count in range(len(self.classes)):
            text_features = self.model.encode_text(self.tokenized_classes[count])
            
            # target class Normalization
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # Dot-product
            similarities = image_features @ text_features.T
            
            probs = (100.0 * similarities).softmax(dim=-1)
            max_idx = probs.argmax(dim=1)
            classes_probs_argmax.append(max_idx)
            
        return classes_probs_argmax
    
    def evaluate(self, dataloader: DataLoader):
        # correct, total = 0, 0
        total = 0
        correct_list: List[int] = [0 for _ in range(len(self.tokenized_classes))]
        # print(len(correct_list))

        with tqdm(total=len(dataloader), unit="Zero-Shot Batches") as progress_bar:

            for images, labels in dataloader:
                images, labels = images.to(self.config.device), labels.to(self.config.device)
                
                predictions = self.predict(images)
                for count, preds in enumerate(predictions, start=0):
                    # print(count)
                    correct_list[count] += (preds == labels).sum().item()
                total += labels.size(0)

                progress_bar.update(1)
        for count, target_class in enumerate(correct_list, start=0):
            print(f'Zero-Shot accuracy for (ex: {self.classes[count][0]}) is: {((target_class / total) * 100):.2f}')
        return
    
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

def prepare_dataloaders(config: Config, resolution: int) -> Tuple[DataLoader, DataLoader]:
    train_set = MistDataSet(config=config, train=True, model_resolution=resolution)
    test_set = MistDataSet(config=config, train=False, model_resolution=resolution)
    train_loader = DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    test_loader = DataLoader(test_set, shuffle=False, batch_size=config.batch_size)
    return train_loader, test_loader        
            
def main():
    conf_dict = {
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        'model_name': 'ViT-B/16',
        'batch_size': 500,
        'linear_probe_c': 0.316,
        'linear_probe_max_iter': 1000,
        'dataset_root': './data'
    }
    
    config = Config.from_dict(conf_dict)

    clip_model, preprocess = clip.load(config.model_name, device=config.device)
    model_resolution = clip_model.visual.input_resolution

    train_loader, test_loader = prepare_dataloaders(config, model_resolution)

    print("---evaluate Zero-Shot on original target names---")
    classes = [
        [str(i) for i in range(10)],
        [f'this is a image of {str(i)}' for i in range(10)],
        [f'there is a {str(i)} digit in a picture' for i in range(10)],
        [f'the picture of handwritten of {str(i)}' for i in range(10)],
    ]
    zer_shot = ZeroShotClassification(config, clip_model, classes)
    zer_shot.evaluate(train_loader)

    # print("---evaluate Zero-Shot on descriptive of target names---")
    # classes = [f'this is a image of {str(i)}' for i in range(10)]
    # zer_shot = ZeroShotClassification(config, clip_model, classes)
    # zer_shot.evaluate(train_loader)

    # print("---evaluate Zero-Shot on descriptive V2 target names---")
    # classes = [f'there is a {str(i)} digit in a picture' for i in range(10)]
    # zer_shot = ZeroShotClassification(config, clip_model, classes)
    # zer_shot.evaluate(train_loader)
    
    # print("---evaluate Zero-Shot with handwritten target names---")
    # classes = [f'the picture of handwritten of {str(i)}' for i in range(10)]
    # zer_shot = ZeroShotClassification(config, clip_model, classes)
    # zer_shot.evaluate(train_loader)

    print("---evaluate Linear Probe---")
    linear_probe = LinearProbe(config, clip_model)
    linear_probe.train_and_evaluate(train_loader, test_loader)

if __name__ == '__main__':
    main()