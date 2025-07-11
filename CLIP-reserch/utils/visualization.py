from typing import List
import matplotlib.pyplot as plt

def visualize(
        class_variations: List[List[str]],
        zero_shot_accuracies: List[int],
        linear_probing_accuracy: int,
        save: bool
    ):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.bar(range(len(class_variations)), zero_shot_accuracies)
    plt.axhline(y=linear_probing_accuracy, color='r', linestyle='--', label='Linear Probe')
    plt.xticks(range(len(class_variations)), [f"ex: {cls[0]}" for cls in class_variations], rotation=90)
    plt.ylabel("Accuracy")
    plt.title("Zero-Shot Accuracies amoung classes variations")
    plt.legend()

    plt.subplot(1, 2, 2)
    max_acc = [max(zero_shot_accuracies), linear_probing_accuracy]
    plt.bar(['Best Zero-Shot acc', 'Linear Probe'], max_acc, color=['blue', 'red'])
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    
    if save:
        plt.savefig("report.png")
        print("\nResults saved to report.png")