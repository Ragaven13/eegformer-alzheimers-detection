import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.data.band_eeg_dataset import BandEEGDataset


BATCH_SIZE = 16


def get_band_dataloaders(batch_size=BATCH_SIZE):
    train_dataset = BandEEGDataset(split="train")
    val_dataset = BandEEGDataset(split="val")
    test_dataset = BandEEGDataset(split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader, test_loader


def main():
    train_loader, val_loader, test_loader = get_band_dataloaders()

    x, y = next(iter(train_loader))

    print("=" * 60)
    print("BAND EEG DATALOADER TEST")
    print("=" * 60)

    print("Batch X shape:", x.shape)
    print("Batch y shape:", y.shape)
    print("First labels:", y[:10])


if __name__ == "__main__":
    main()