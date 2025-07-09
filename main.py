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

class MistDataSet(Dataset):
    def __init__(
            self,
            root: str = "./data",
            train: bool = True,
            download: bool = True,
            model_resolution = None
        ):
        self.base_data = torchvision.datasets.MNIST(root=root, train=train, download=download)
        self.images: List[torch.Tensor] = []
        self.model_resolution = model_resolution
        self.transform, self.label_transform = self._get_transforms()
    
    def __len__(self) -> int:
        return len(self.base_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, label = self.base_data[idx]

        if self.transform:
            img = self.transform(img)
        # if self.label_transform:
            # label = self.label_transform(label)
        
        return img, label

    def _get_transforms(self):
        n_px = self.model_resolution
        transform = transforms.Compose([
        transforms.Resize(n_px, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(n_px),
        transforms.Lambda(lambda x: x.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

        label_transform = transforms.Compose([
            transforms.Lambda(lambda label: f'an image of the {label} digit')
        ])
        return transform, label_transform
    

class VLM:
    def __init__(self, classes: List[str]):
        self.classes = classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, self.preprocess = clip.load('ViT-B/16', device=self.device) #preprocess also accept Tensors
        self.tokenized_classes = clip.tokenize(self.classes).to(self.device)
    
    @torch.inference_mode()
    def predict(self, images: torch.Tensor):
        image_features = self.model.encode_image(images)
        text_features = self.model.encode_text(self.tokenized_classes)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return similarity.argmax(dim=1)

    def validation(self, dataloader: DataLoader):
        correct, total = 0, 0

        with tqdm(total=len(dataloader), unit="batch") as progress_bar:
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                classified_img = self.predict(images)
                correct += (classified_img == labels).sum().item()
                total += labels.size(0)
                progress_bar.update(1)
                progress_bar.set_postfix(accuracy=f"{100 * correct/total:5.2f}%")

        print(f'Accuracy - {((correct / total) * 100):.2f}')
        return

if __name__ == '__main__':
    classes = [str(i) for i in range(10)]
    vlm = VLM(classes)

    train_MNIST_dataset = MistDataSet(model_resolution=vlm.model.visual.input_resolution)
    train_loader = DataLoader(train_MNIST_dataset, shuffle=False, batch_size=500)

    vlm.validation(train_loader)