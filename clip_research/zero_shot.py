from typing import List

import clip
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config import Config


class ZeroShotClassification:
    """
    Zero-shot classification using CLIP model.

    Attributes:
    ----------
    config : Config
        Configuration object containing model settings.
    model : torch.nn.Module
        CLIP model.
    classes : List[List[str]]
        List of classes for classification.
    text_features : List[torch.Tensor]
        Pre-computed text features for classes.

    Methods:
    -------
    __init__(config: Config, clip_model: torch.nn.Module, classes: List[List[str]])
        Initializes the zero-shot classification object.
    _text_tokenizer() -> List[torch.Tensor]
        Pre-computes text features for classes.
    predict(images: torch.Tensor) -> List[int]
        Predicts class labels for given images.
    evaluate(dataloader: DataLoader)
        Evaluates the model on a given data loader.
    """

    def __init__(
        self,
        config: Config,
        clip_model: torch.nn.Module,
        classes: List[List[str]],
    ):
        self.config = config
        self.model = clip_model
        self.classes = classes
        self.text_features = self._text_tokenizer()

    @torch.inference_mode()
    def _text_tokenizer(self) -> List[torch.Tensor]:
        """
        Separate class tokenization logic to avoid recomputing text features
        during the evaluation process.
        """
        text_features: List[torch.Tensor] = []

        for target_cls in self.classes:
            tokenized_text = clip.tokenize(target_cls).to(self.config.device)
            encoded_text = self.model.encode_text(tokenized_text)
            encoded_text /= encoded_text.norm(
                dim=-1, keepdim=True
            )  # target class Normalization

            text_features.append(encoded_text)

        return text_features

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> List[int]:
        """
        Predicts class labels for given images.
        """
        classes_probs_argmax: List[int] = []

        image_features = self.model.encode_image(images)
        image_features /= image_features.norm(
            dim=-1, keepdim=True
        )  # Image normalization

        for count in range(len(self.classes)):
            similarities = image_features @ self.text_features[count].T  # Dot-product

            probs = (100.0 * similarities).softmax(dim=-1)
            max_idx = probs.argmax(dim=1)
            classes_probs_argmax.append(max_idx)

        return classes_probs_argmax

    def evaluate(self, dataloader: DataLoader):
        """
        Evaluates the model on a given data loader.

        Returns:
        -------
        List[float]
            List of accuracy scores for each class.
        """
        correct_list = [0 for _ in range(len(self.text_features))]
        total = 0

        with tqdm(total=len(dataloader), unit="Zero-Shot Batches") as progress_bar:

            for images, labels in dataloader:
                images, labels = images.to(self.config.device), labels.to(
                    self.config.device
                )

                predictions = self.predict(images)
                for count, preds in enumerate(predictions, start=0):
                    # print(count)
                    correct_list[count] += (preds == labels).sum().item()
                total += labels.size(0)

                progress_bar.update(1)

        return [(acc / total) * 100 for acc in correct_list]
