import random
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

BEST_CHECKPOINT = Path("esc50_resnet18_best.pth")
LAST_CHECKPOINT = Path("esc50_resnet18_last.pth")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    if not (ROOT / "meta" / "esc50.csv").is_file():
        raise FileNotFoundError(
            "Cannot find ESC-50/meta/esc50.csv. "
            "Run this script from the project root."
        )

    set_seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    train_dataset = ESC50(str(ROOT), [1, 2, 3, 4])
    val_dataset = ESC50(str(ROOT), [5])

    generator = torch.Generator().manual_seed(SEED)

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

    print("Training clips:", len(train_dataset))
    print("Validation clips:", len(val_dataset))

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

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_train = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)
            loss = loss_fn(logits, y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            total_train += y.size(0)

        train_loss = total_loss / total_train

        model.eval()
        correct = 0
        total_val = 0

        with torch.inference_mode():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                predictions = model(x).argmax(dim=1)
                correct += (predictions == y).sum().item()
                total_val += y.size(0)

        final_accuracy = correct / total_val

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_accuracy={final_accuracy:.4f} "
            f"({final_accuracy * 100:.2f}%)"
        )

        if final_accuracy > best_accuracy:
            best_accuracy = final_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                BEST_CHECKPOINT,
            )

    torch.save(model.state_dict(), LAST_CHECKPOINT)

    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Final validation accuracy: {final_accuracy:.4f}")
    print("Best checkpoint:", BEST_CHECKPOINT)
    print("Final checkpoint:", LAST_CHECKPOINT)


if __name__ == "__main__":
    main()