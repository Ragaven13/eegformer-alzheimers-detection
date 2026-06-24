from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


PROCESSED_DIR = Path("data/processed")
BAND_WINDOW_DIR = PROCESSED_DIR / "band_windows"


class BandEEGDataset(Dataset):
    """
    PyTorch Dataset for EEGFormer.

    Each sample:

        X shape: (5, 19, 2500)

    Meaning:

        5    = frequency bands
        19   = EEG electrodes
        2500 = 5-second EEG window samples

    Bands:

        0 = Delta
        1 = Theta
        2 = Alpha
        3 = Beta
        4 = Gamma

    Labels:

        0 = Alzheimer's
        1 = Healthy Control
        2 = Frontotemporal Dementia
    """

    def __init__(self, split):
        if split not in ["train", "val", "test"]:
            raise ValueError("split must be one of: train, val, test")

        self.split = split

        split_subjects = np.load(
            PROCESSED_DIR / f"{split}_subjects.npy",
            allow_pickle=True,
        )

        self.samples = []

        for subject_id in split_subjects:
            x_path = BAND_WINDOW_DIR / f"{subject_id}_X_band.npy"
            y_path = BAND_WINDOW_DIR / f"{subject_id}_y.npy"

            if not x_path.exists():
                raise FileNotFoundError(f"Missing file: {x_path}")

            if not y_path.exists():
                raise FileNotFoundError(f"Missing file: {y_path}")

            X_subject = np.load(x_path, mmap_mode="r")
            y_subject = np.load(y_path)

            for index in range(len(y_subject)):
                self.samples.append(
                    {
                        "x_path": x_path,
                        "y": int(y_subject[index]),
                        "index": index,
                        "subject_id": subject_id,
                    }
                )

        print(f"{split.upper()} Band EEG dataset loaded")
        print(f"Samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        X_subject = np.load(
            sample["x_path"],
            mmap_mode="r",
        )

        x = X_subject[sample["index"]]
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
    train_dataset = BandEEGDataset(split="train")

    x, y = train_dataset[0]

    print("\nSingle sample check")
    print("X shape:", x.shape)
    print("y:", y)