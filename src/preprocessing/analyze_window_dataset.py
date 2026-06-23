from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data/raw/ds004504")
PROCESSED_DIR = Path("data/processed")

LABEL_NAMES = {
    0: "Alzheimer's",
    1: "Healthy Control",
    2: "Frontotemporal Dementia",
}


def main():
    y = np.load(PROCESSED_DIR / "y_labels.npy")
    subject_ids = np.load(PROCESSED_DIR / "subject_ids.npy", allow_pickle=True)
    participants = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")

    print("=" * 60)
    print("WINDOW DATASET ANALYSIS")
    print("=" * 60)

    print(f"\nTotal windows: {len(y)}")
    print(f"Total unique subjects: {len(np.unique(subject_ids))}")

    print("\nWindow count per class:")
    for label, class_name in LABEL_NAMES.items():
        count = np.sum(y == label)
        print(f"{label} | {class_name}: {count}")

    print("\nSubject count per group:")
    print(participants["Group"].value_counts())

    print("\nWindows per subject:")
    unique_subjects, counts = np.unique(subject_ids, return_counts=True)

    for subject, count in zip(unique_subjects[:10], counts[:10]):
        print(f"{subject}: {count}")

    print("\nMin windows per subject:", counts.min())
    print("Max windows per subject:", counts.max())
    print("Mean windows per subject:", counts.mean())


if __name__ == "__main__":
    main()