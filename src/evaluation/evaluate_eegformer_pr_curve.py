import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

from src.data.band_dataloader import get_band_dataloaders
from src.models.eegformer import EEGFormer


CHECKPOINT_PATH = "results/checkpoints/eegformer.pt"
FIGURE_DIR = Path("results/figures")

LABEL_NAMES = ["Alzheimer's", "Healthy Control", "FTD"]
NUM_CLASSES = 3


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("=" * 60)
    print("EEGFORMER PRECISION-RECALL EVALUATION")
    print("=" * 60)
    print("Device:", device)

    _, _, test_loader = get_band_dataloaders(batch_size=16)

    model = EEGFormer().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.numpy())

    y_score = np.concatenate(all_probs, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])

    plt.figure(figsize=(8, 6))

    for class_index in range(NUM_CLASSES):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, class_index],
            y_score[:, class_index],
        )

        ap = average_precision_score(
            y_true_bin[:, class_index],
            y_score[:, class_index],
        )

        plt.plot(
            recall,
            precision,
            label=f"{LABEL_NAMES[class_index]} AP = {ap:.3f}",
        )

        print(f"{LABEL_NAMES[class_index]} Average Precision: {ap:.4f}")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("EEGFormer Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()

    output_path = FIGURE_DIR / "eegformer_precision_recall_curve.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"\nSaved PR curve to: {output_path}")


if __name__ == "__main__":
    main()