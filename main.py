import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import clip
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MistDataSet(Dataset):
    def __init__(
            self,
            root: str = "./data",
            train: bool = True,
            download: bool = True,
            model_resolution = None
        ):
        self.base_data = torchvision.datasets.MNIST(root=root, train=train, download=download)
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
    def __init__(self, clip_model: torch.nn.Module, classes: List[str]):
        self.model = clip_model
        self.classes = classes
        self.tokenized_classes = clip.tokenize(self.classes).to(DEVICE)
    
    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(self.tokenized_classes)
        
        # Normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Dot-product
        similarities = image_features @ text_features.T
        
        probs = (100.0 * similarities).softmax(dim=-1)
        return probs.argmax(dim=1)

    def evaluate(self, dataloader: DataLoader):
        correct, total = 0, 0

        with tqdm(total=len(dataloader), unit="Zero-Shot Batches") as progress_bar:

            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                predictions = self.predict(images)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                progress_bar.update(1)

        print(f'Zero-Shot accuracy: {((correct / total) * 100):.2f}')
        return
    
class LinearProbe:
    def __init__(self, clip_model: torch.nn.Module):
        self.model = clip_model
    
    @torch.inference_mode()
    def extract_features(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []

        with tqdm(total = len(dataloader), unit='Feature Extraction') as progress_bar:
            for images, labels in dataloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)

                features = self.model.encode_image(images)
                all_features.append(features)
                all_labels.append(labels)

                progress_bar.update(1)

        return torch.cat(all_features).cpu().numpy(), torch.cat(all_labels).cpu().numpy()
        
    def train_and_evaluate(self, train_loader: DataLoader, test_loader: DataLoader):
        train_features, train_labels = self.extract_features(train_loader)
        test_features, test_labels = self.extract_features(test_loader)

        clf = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
        clf.fit(train_features, train_labels)

        preds = clf.predict(test_features)
        accuracy = accuracy_score(test_labels, preds) * 100
        print(f"Linear Probe accuracy = {accuracy:.2f}")

        
            
def main():
    classes = [str(i) for i in range(10)]

    clip_model, preprocess = clip.load('ViT-B/16', device=DEVICE)
    model_resolution = clip_model.visual.input_resolution


    train_MNIST_dataset = MistDataSet(model_resolution=model_resolution)
    test_MNIST_dataset = MistDataSet(model_resolution=model_resolution)

    train_loader = DataLoader(train_MNIST_dataset, shuffle=True, batch_size=500)
    test_loader = DataLoader(test_MNIST_dataset, shuffle=False, batch_size=500)

    zer_shot = ZeroShotClassification(clip_model, classes)
    zer_shot.evaluate(train_loader)

    linear_probe = LinearProbe(clip_model)
    linear_probe.train_and_evaluate(train_loader, test_loader)

if __name__ == '__main__':
    main()