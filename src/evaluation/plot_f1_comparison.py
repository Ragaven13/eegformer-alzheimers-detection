from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_TABLE = Path("results/tables/model_comparison.csv")
FIGURE_DIR = Path("results/figures")


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RESULTS_TABLE)

    plt.figure(figsize=(10, 6))
    plt.bar(df["Model"], df["Macro F1"])
    plt.ylim(0, 1.0)
    plt.ylabel("Test Macro F1")
    plt.title("Model Comparison - Macro F1")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    output_path = FIGURE_DIR / "model_f1_comparison.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved figure: {output_path}")


if __name__ == "__main__":
    main()