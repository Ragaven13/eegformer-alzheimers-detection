import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import Dataset


PROCESSED_DIR = Path("data/processed")
BAND_WINDOW_DIR = PROCESSED_DIR / "band_windows"

BAND_NAMES = {
    0: "delta",
    1: "theta",
    2: "alpha",
    3: "beta",
    4: "gamma",
}


class SingleBandEEGDataset(Dataset):
    def __init__(self, split, band_index):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be train, val, or test")

        if band_index not in BAND_NAMES:
            raise ValueError("band_index must be 0, 1, 2, 3, or 4")

        self.split = split
        self.band_index = band_index
        self.band_name = BAND_NAMES[band_index]

        split_subjects = np.load(
            PROCESSED_DIR / f"{split}_subjects.npy",
            allow_pickle=True,
        )

        self.samples = []

        for subject_id in split_subjects:
            x_path = BAND_WINDOW_DIR / f"{subject_id}_X_band.npy"
            y_path = BAND_WINDOW_DIR / f"{subject_id}_y.npy"

            X_subject = np.load(x_path, mmap_mode="r")
            y_subject = np.load(y_path)

            for index in range(len(y_subject)):
                self.samples.append(
                    {
                        "x_path": x_path,
                        "index": index,
                        "y": int(y_subject[index]),
                    }
                )

        print(f"{split.upper()} Single-Band Dataset Loaded")
        print(f"Band: {self.band_name}")
        print(f"Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        X_subject = np.load(sample["x_path"], mmap_mode="r")

        x = X_subject[sample["index"], self.band_index]
        y = sample["y"]

        x = x.astype("float32")

        mean = x.mean()
        std = x.std()

        if std > 0:
            x = (x - mean) / std

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y


if __name__ == "__main__":
    dataset = SingleBandEEGDataset(split="train", band_index=2)

    x, y = dataset[0]

    print("\nSingle sample check")
    print("Band:", dataset.band_name)
    print("X shape:", x.shape)
    print("y:", y)