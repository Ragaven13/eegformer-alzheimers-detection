from torch.utils.data import DataLoader

from src.data.eeg_dataset import EEGWindowDataset

BATCH_SIZE = 32


def get_dataloaders(batch_size=BATCH_SIZE):

    train_dataset = EEGWindowDataset(split="train")
    val_dataset = EEGWindowDataset(split="val")
    test_dataset = EEGWindowDataset(split="test")

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

    train_loader, val_loader, test_loader = get_dataloaders()

    print("=" * 60)
    print("DATALOADER TEST")
    print("=" * 60)

    x, y = next(iter(train_loader))

    print("\nBatch Shape:")
    print(x.shape)

    print("\nLabel Shape:")
    print(y.shape)

    print("\nFirst Batch Labels:")
    print(y[:10])


if __name__ == "__main__":
    main()