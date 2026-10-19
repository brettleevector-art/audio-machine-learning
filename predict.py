
import argparse
from pathlib import Path

import pandas as pd
import torch

from dataset import load_audio_spectrogram
from resnet_model import build_resnet18


ROOT = Path("ESC-50")
CHECKPOINT = Path("esc50_resnet18_best.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_class_names():
    metadata = pd.read_csv(ROOT / "meta" / "esc50.csv")
    rows = metadata[["target", "category"]].drop_duplicates()

    names = {
        int(target): str(category)
        for target, category in rows.itertuples(
            index=False,
            name=None,
        )
    }

    if set(names) != set(range(50)):
        raise ValueError(
            "ESC-50 metadata must map class IDs 0 through 49."
        )

    return names


def load_model():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint: {CHECKPOINT}. "
            "Check that the trained ResNet18 weights are available."
        )

    model = build_resnet18(pretrained=False)

    state_dict = torch.load(
        CHECKPOINT,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    return model


CLASS_NAMES = load_class_names()
MODEL = load_model()


def predict_audio(audio_path):
    if not audio_path:
        raise ValueError("Provide an audio file.")

    if not Path(audio_path).is_file():
        raise FileNotFoundError(
            f"Audio file not found: {audio_path}"
        )

    x = load_audio_spectrogram(audio_path).unsqueeze(0)
    x = x.to(DEVICE)

    with torch.inference_mode():
        logits = MODEL(x)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_probs, top_indices = torch.topk(probabilities, k=5)

    return {
        CLASS_NAMES[index.item()]: float(probability.item())
        for probability, index in zip(top_probs, top_indices)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_path",
        help="Path to an audio file",
    )
    args = parser.parse_args()

    for category, probability in predict_audio(args.audio_path).items():
        print(f"{category}: {probability * 100:.2f}%")
