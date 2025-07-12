from typing import List
import matplotlib.pyplot as plt

def visualize(
        class_variations: List[List[str]],
        zero_shot_accuracies: List[int],
        linear_probing_accuracy: int,
        save: bool
    ) -> None:
    """
    Visualize zero-shot accuracies among class variations and compare with linear probing accuracy.

    Parameters:
    ----------
    class_variations : List[List[str]]
        List of class name variations.
    zero_shot_accuracies : List[int]
        List of zero-shot accuracies corresponding to class variations.
    linear_probing_accuracy : int
        Linear probing accuracy.
    save : bool
        Flag to save the plot.

    Returns:
    -------
    None
    """
    plt.figure(figsize=(12, 6))

    # comaring zero-shot accuracies for diffrent class names with linear probe accuracy
    plt.subplot(1, 2, 1)
    plt.bar(range(len(class_variations)), zero_shot_accuracies)
    plt.axhline(y=linear_probing_accuracy, color='r', linestyle='--', label='Linear Probe')
    plt.xticks(range(len(class_variations)), [f"ex: {cls[0]}" for cls in class_variations], rotation=90)
    plt.ylabel("Accuracy")
    plt.title("Zero-Shot Accuracies amoung classes variations")
    plt.legend()

    # comparing maximum zero-shot accuracy (among all of the class name variations) and linear probe
    plt.subplot(1, 2, 2)
    max_acc = [max(zero_shot_accuracies), linear_probing_accuracy]
    plt.bar(['Best Zero-Shot acc', 'Linear Probe'], max_acc, color=['blue', 'red'])
    plt.ylabel("Accuracy")
    
    plt.tight_layout()
    
    if save:
        plt.savefig("report.png")
        print("\nResults saved to report.png")