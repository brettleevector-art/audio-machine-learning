from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dataset import ESC50
from model import Net
from resnet_model import build_resnet18


ROOT = "ESC-50"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_checkpoint(name, checkpoint_path, model_factory, loader):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint_path}"
        )

    model = model_factory().to(DEVICE)

    state_dict = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            predictions = model(x).argmax(dim=1)

            correct += (predictions == y).sum().item()
            total += y.size(0)

    if total == 0:
        raise ValueError("Validation dataset is empty.")

    accuracy = correct / total

    print(
        f"{name}: {correct}/{total} correct | "
        f"accuracy={accuracy:.4f} ({accuracy * 100:.2f}%)"
    )

    return accuracy


def main():
    val_dataset = ESC50(ROOT, [5])

    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    experiments = [
        (
            "Baseline CNN",
            "esc50_cnn.pth",
            Net,
        ),
        (
            "CNN + SpecAugment",
            "esc50_cnn_specaugment.pth",
            Net,
        ),
        (
            "ResNet18",
            "esc50_resnet18_last.pth",
            lambda: build_resnet18(pretrained=False),
        ),
    ]

    print("Validation fold: 5")
    print("Validation clips:", len(val_dataset))

    results = []

    for name, checkpoint_path, model_factory in experiments:
        accuracy = evaluate_checkpoint(
            name,
            checkpoint_path,
            model_factory,
            val_loader,
        )
        results.append((name, accuracy))

    # Prepare chart data.
    labels = [name for name, _ in results]
    values = [accuracy * 100 for _, accuracy in results]

    # Create horizontal bar chart.
    fig, ax = plt.subplots(figsize=(9, 4.5))

    bars = ax.barh(labels, values)

    ax.set_xlim(0, 100)
    ax.set_xlabel("Validation accuracy (%)")
    ax.set_title("ESC-50 Model Comparison — Fold 5")
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)

    # Display accuracy beside each bar.
    for bar, value in zip(bars, values):
        ax.text(
            value + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}%",
            va="center",
        )

    fig.tight_layout()

    # Save the chart in the project folder.
    output_path = "model_comparison_fold5.png"

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Chart saved to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()