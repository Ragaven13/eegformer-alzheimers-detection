from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ABLATION_TABLE = Path("results/tables/ablation_study.csv")
FIGURE_DIR = Path("results/figures")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(ABLATION_TABLE)

    plt.figure(figsize=(11, 6))
    plt.bar(df["Experiment"], df["Macro F1"])
    plt.ylim(0, 1.0)
    plt.ylabel("Test Macro F1")
    plt.title("EEGFormer Ablation Study")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    output_path = FIGURE_DIR / "eegformer_ablation_study.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()