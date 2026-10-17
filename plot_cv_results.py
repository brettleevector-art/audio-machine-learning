import json
from pathlib import Path

import matplotlib.pyplot as plt


RESULTS_PATH = Path("resnet18_cv_results.json")
OUTPUT_PATH = Path("resnet18_five_fold_cv.png")


def main():
    if not RESULTS_PATH.is_file():
        raise FileNotFoundError(
            "Run cross_validate_resnet18.py first."
        )

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    fold_results = data.get("fold_results", [])

    if data.get("completed_folds") != 5 or len(fold_results) != 5:
        raise ValueError(
            "The results file must contain all five completed folds."
        )

    fold_results = sorted(
        fold_results,
        key=lambda result: result["validation_fold"],
    )

    fold_numbers = [
        result["validation_fold"] for result in fold_results
    ]

    if fold_numbers != [1, 2, 3, 4, 5]:
        raise ValueError(
            f"Unexpected validation folds: {fold_numbers}"
        )

    accuracies = [
        result["final_accuracy"] * 100
        for result in fold_results
    ]

    mean_accuracy = data["mean_final_accuracy"] * 100
    std_accuracy = data["std_final_accuracy"] * 100

    labels = [f"Fold {fold}" for fold in fold_numbers]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, accuracies)

    ax.axhline(
        mean_accuracy,
        linestyle="--",
        label=f"Mean accuracy: {mean_accuracy:.2f}%",
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel("Validation fold")
    ax.set_ylabel("Final validation accuracy (%)")
    ax.set_title("WILDLIGHT — ResNet18 Five-Fold Evaluation")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            accuracy + 1,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom",
        )

    ax.legend()

    fig.text(
        0.99,
        0.01,
        f"Mean: {mean_accuracy:.2f}% | "
        f"Std. deviation: {std_accuracy:.2f} percentage points",
        ha="right",
        fontsize=9,
    )

    fig.tight_layout(rect=(0, 0.05, 1, 1))

    fig.savefig(
        OUTPUT_PATH,
        dpi=200,
        bbox_inches="tight",
    )

    print("Chart saved to:", OUTPUT_PATH)
    print(f"Mean final accuracy: {mean_accuracy:.2f}%")
    print(f"Standard deviation: {std_accuracy:.2f} percentage points")

    plt.show()


if __name__ == "__main__":
    main()