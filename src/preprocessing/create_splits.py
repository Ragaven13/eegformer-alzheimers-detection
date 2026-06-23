from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data/raw/ds004504")
PROCESSED_DIR = Path("data/processed")

RANDOM_SEED = 42

GROUP_TO_LABEL = {
    "A": 0,
    "C": 1,
    "F": 2,
}


def load_subject_labels():
    participants = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")

    subjects = participants["participant_id"].values
    labels = participants["Group"].map(GROUP_TO_LABEL).values

    return subjects, labels


def main():
    subjects, labels = load_subject_labels()

    train_subjects, temp_subjects, train_labels, temp_labels = train_test_split(
        subjects,
        labels,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    val_subjects, test_subjects, val_labels, test_labels = train_test_split(
        temp_subjects,
        temp_labels,
        test_size=0.50,
        random_state=RANDOM_SEED,
        stratify=temp_labels,
    )

    np.save(PROCESSED_DIR / "train_subjects.npy", train_subjects)
    np.save(PROCESSED_DIR / "val_subjects.npy", val_subjects)
    np.save(PROCESSED_DIR / "test_subjects.npy", test_subjects)

    print("=" * 60)
    print("SUBJECT-LEVEL SPLIT CREATED")
    print("=" * 60)

    print(f"Train subjects: {len(train_subjects)}")
    print(f"Validation subjects: {len(val_subjects)}")
    print(f"Test subjects: {len(test_subjects)}")

    print("\nTrain label distribution:")
    print(pd.Series(train_labels).value_counts().sort_index())

    print("\nValidation label distribution:")
    print(pd.Series(val_labels).value_counts().sort_index())

    print("\nTest label distribution:")
    print(pd.Series(test_labels).value_counts().sort_index())

    print("\nSaved split files:")
    print(PROCESSED_DIR / "train_subjects.npy")
    print(PROCESSED_DIR / "val_subjects.npy")
    print(PROCESSED_DIR / "test_subjects.npy")


if __name__ == "__main__":
    main()