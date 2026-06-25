import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
import matplotlib.pyplot as plt

from src.data.band_dataloader import get_band_dataloaders
from src.models.eegformer import EEGFormer


CHECKPOINT_PATH = "results/checkpoints/eegformer.pt"
FIGURE_DIR = Path("results/figures")

BAND_NAMES = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]


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
    print("CROSS-BAND ATTENTION VISUALIZATION")
    print("=" * 60)
    print("Device:", device)

    _, _, test_loader = get_band_dataloaders(batch_size=16)

    model = EEGFormer().to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT_PATH,
            map_location=device,
        )
    )

    model.eval()

    attention_matrices = []

    with torch.no_grad():
        for batch_index, (x, y) in enumerate(test_loader):
            x = x.to(device)

            _ = model(x)

            attention = model.get_attention_weights()

            # attention shape:
            # (batch, heads, bands, bands)
            attention = attention.detach().cpu().numpy()

            # average across batch and heads
            attention = attention.mean(axis=(0, 1))

            attention_matrices.append(attention)

            if batch_index >= 20:
                break

    mean_attention = np.mean(
        attention_matrices,
        axis=0,
    )

    print("\nMean Cross-Band Attention Matrix:")
    print(mean_attention)

    plt.figure(figsize=(7, 6))

    plt.imshow(
        mean_attention,
        aspect="auto",
    )

    plt.colorbar(label="Attention Weight")

    plt.xticks(
        ticks=np.arange(len(BAND_NAMES)),
        labels=BAND_NAMES,
        rotation=45,
        ha="right",
    )

    plt.yticks(
        ticks=np.arange(len(BAND_NAMES)),
        labels=BAND_NAMES,
    )

    plt.xlabel("Attended Band")
    plt.ylabel("Query Band")
    plt.title("EEGFormer Cross-Band Attention Heatmap")

    for i in range(len(BAND_NAMES)):
        for j in range(len(BAND_NAMES)):
            plt.text(
                j,
                i,
                f"{mean_attention[i, j]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output_path = FIGURE_DIR / "cross_band_attention_heatmap.png"

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    print(f"\nSaved attention heatmap to: {output_path}")


if __name__ == "__main__":
    main()