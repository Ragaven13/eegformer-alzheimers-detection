from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


PROCESSED_DIR = Path("data/processed")


class EEGWindowDataset(Dataset):
    """
    PyTorch Dataset for EEG window classification.

    This dataset loads preprocessed EEG windows and returns one window
    with its diagnosis label at a time.

    Each sample has shape:

        X: (19, 2500)
        y: class label

    Labels:

        0 = Alzheimer's Disease
        1 = Healthy Control
        2 = Frontotemporal Dementia
    """

    def __init__(self, split):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of: train, val, test")

        self.split = split

        self.X = np.load(PROCESSED_DIR / "X_windows.npy")
        self.y = np.load(PROCESSED_DIR / "y_labels.npy")
        self.subject_ids = np.load(
            PROCESSED_DIR / "subject_ids.npy",
            allow_pickle=True
        )

        split_subjects = np.load(
            PROCESSED_DIR / f"{split}_subjects.npy",
            allow_pickle=True
        )

        mask = np.isin(self.subject_ids, split_subjects)

        self.X = self.X[mask]
        self.y = self.y[mask]
        self.subject_ids = self.subject_ids[mask]

        print(f"{split.upper()} dataset loaded")
        print(f"Windows: {len(self.X)}")
        print(f"X shape: {self.X.shape}")
        print(f"y shape: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        window = self.X[index]
        label = self.y[index]

        window = torch.tensor(window, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        return window, label


if __name__ == "__main__":
    train_dataset = EEGWindowDataset(split="train")
    val_dataset = EEGWindowDataset(split="val")
    test_dataset = EEGWindowDataset(split="test")

    x, y = train_dataset[0]

    print("\nSingle sample check")
    print("X shape:", x.shape)
    print("y:", y)