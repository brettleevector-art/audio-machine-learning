
import json
import random
import statistics
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import ESC50
from resnet_model import build_resnet18


ROOT = Path("ESC-50")
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
SEED = 42

RESULTS_FILE = Path("resnet18_cv_results.json")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(validation_fold, device):
    seed = SEED + validation_fold
    set_seed(seed)

    training_folds = [
        fold
        for fold in range(1, 6)
        if fold != validation_fold
    ]

    train_dataset = ESC50(str(ROOT), training_folds)
    val_dataset = ESC50(str(ROOT), [validation_fold])

    if len(train_dataset) != 1600:
        raise ValueError(
            f"Fold {validation_fold}: expected 1600 training "
            f"clips, found {len(train_dataset)}."
        )

    if len(val_dataset) != 400:
        raise ValueError(
            f"Fold {validation_fold}: expected 400 validation "
            f"clips, found {len(val_dataset)}."
        )

    generator = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = build_resnet18(pretrained=True).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    loss_fn = torch.nn.CrossEntropyLoss()

    best_accuracy = -1.0
    best_epoch = 0
    final_accuracy = 0.0

    print(
        f"\nValidation fold: {validation_fold} | "
        f"Training folds: {training_folds}"
    )

    for epoch in range(EPOCHS):
        model.train()

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0

        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x).argmax(dim=1)
                correct += (predictions == y).sum().item()
                total += y.size(0)

        final_accuracy = correct / total

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Validation accuracy: {final_accuracy:.4f} "
            f"({final_accuracy * 100:.2f}%)"
        )

        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_epoch = epoch + 1

    result = {
        "validation_fold": validation_fold,
        "training_folds": training_folds,
        "training_clips": len(train_dataset),
        "validation_clips": len(val_dataset),
        "seed": seed,
        "final_accuracy": final_accuracy,
        "best_accuracy": best_accuracy,
        "best_epoch": best_epoch,
    }

    print(
        f"Fold {validation_fold} complete | "
        f"Final: {final_accuracy * 100:.2f}% | "
        f"Best: {best_accuracy * 100:.2f}% "
        f"(epoch {best_epoch})"
    )

    return result


def main():
    if not (ROOT / "meta" / "esc50.csv").is_file():
        raise FileNotFoundError(
            "Cannot find ESC-50/meta/esc50.csv. "
            "Run this script from the Audio_Machine_Learning "
            "project root."
        )

    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Model: pretrained ResNet18")
    print("Device:", device)
    print("Epochs per fold:", EPOCHS)
    print("Validation protocol: official five-fold CV")

    fold_results = []

    for validation_fold in range(1, 6):
        result = train_one_fold(validation_fold, device)
        fold_results.append(result)

        # Save completed fold results immediately.
        # If a later fold fails, completed results are preserved.
        final_scores = [
            item["final_accuracy"] for item in fold_results
        ]

        summary = {
            "dataset": "ESC-50",
            "model": "Pretrained ResNet18",
            "seed_base": SEED,
            "epochs_per_fold": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "completed_folds": len(fold_results),
            "fold_results": fold_results,
            "mean_final_accuracy": statistics.mean(final_scores),
            "std_final_accuracy": (
                statistics.pstdev(final_scores)
                if len(final_scores) > 1
                else 0.0
            ),
        }

        RESULTS_FILE.write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

    final_scores = [
        item["final_accuracy"] for item in fold_results
    ]

    print("\nFive-fold summary")
    for result in fold_results:
        print(
            f"Fold {result['validation_fold']}: "
            f"{result['final_accuracy'] * 100:.2f}%"
        )

    print(
        f"Mean final accuracy: "
        f"{statistics.mean(final_scores) * 100:.2f}%"
    )

    print(
        f"Standard deviation: "
        f"{statistics.pstdev(final_scores) * 100:.2f} "
        "percentage points"
    )

    print("Results saved to:", RESULTS_FILE)


if __name__ == "__main__":
    main()
