from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

FIGURE_DIR = Path("results/figures")

LABEL_NAMES = ["Alzheimer's", "Healthy", "FTD"]

CONFUSION_MATRIX = np.array([
    [749, 164, 104],
    [170, 688, 21],
    [161, 53, 237],
])


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    display = ConfusionMatrixDisplay(
        confusion_matrix=CONFUSION_MATRIX,
        display_labels=LABEL_NAMES,
    )

    display.plot(values_format="d")
    plt.title("EEGFormer Confusion Matrix")
    plt.tight_layout()

    output_path = FIGURE_DIR / "eegformer_confusion_matrix.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()