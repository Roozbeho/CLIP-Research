from typing import List

import clip
from tqdm import tqdm
from utils.config import Config
import torch
from torch.utils.data import DataLoader

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
        correct_list = [0 for _ in range(len(self.tokenized_classes))]
        total = 0

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