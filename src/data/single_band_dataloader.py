import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from torch.utils.data import DataLoader
from src.data.single_band_dataset import SingleBandEEGDataset


def get_single_band_dataloaders(band_index, batch_size=16):
    train_dataset = SingleBandEEGDataset("train", band_index)
    val_dataset = SingleBandEEGDataset("val", band_index)
    test_dataset = SingleBandEEGDataset("test", band_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, _, _ = get_single_band_dataloaders(band_index=2)

    x, y = next(iter(train_loader))

    print("Batch X shape:", x.shape)
    print("Batch y shape:", y.shape)