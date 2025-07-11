# TODO: improve performance: the code recompute text feature for each batch
# TODO: Add visualization of accuracies
# TODO: Add more class variations
# TODO: change models outputs
# TODO: Add docs and comment in code for readability

import clip

from linear_probe import LinearProbe
from zero_shot import ZeroShotClassification
from utils.config import Config
from data_loader import prepare_dataloaders
import torch


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

    print("Loading and prepairing data")
    train_loader, test_loader = prepare_dataloaders(config, model_resolution)

    print("="*50)
    print("Evaluating Zero-Shot Performance")
    classes = [
        [str(i) for i in range(10)],
        [f'this is a image of {str(i)}' for i in range(10)],
        [f'there is a {str(i)} digit in a picture' for i in range(10)],
        [f'the picture of handwritten of {str(i)}' for i in range(10)],
    ]

    zero_shot = ZeroShotClassification(config, clip_model, classes)
    zero_shot_accuracies = zero_shot.evaluate(train_loader)

    print("Zero-Shot accuracies:")
    for i, acc in enumerate(zero_shot_accuracies):
        print(f"(ex: {classes[i][0]}) : {acc:.2f}")

    print("="*50)
    print("Evaluating Linear Probe Performance")
    linear_probe = LinearProbe(config, clip_model)
    linear_probe_accuracies = linear_probe.train_and_evaluate(train_loader, test_loader)

    print(f"Linear Probe accuracies : {linear_probe_accuracies:.2f}")

if __name__ == '__main__':
    main()