import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score

from src.data.single_band_dataloader import get_single_band_dataloaders
from src.data.single_band_dataset import BAND_NAMES
from src.models.single_band_eegformer import SingleBandEEGFormer


EPOCHS = 10
LEARNING_RATE = 1e-4


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    preds_all = []
    labels_all = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")

    return total_loss / len(loader), acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)

            preds_all.extend(preds.cpu().numpy())
            labels_all.extend(y.cpu().numpy())

    acc = accuracy_score(labels_all, preds_all)
    f1 = f1_score(labels_all, preds_all, average="macro")

    return total_loss / len(loader), acc, f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", type=int, required=True, choices=[0, 1, 2, 3, 4])
    args = parser.parse_args()

    band_index = args.band
    band_name = BAND_NAMES[band_index]

    checkpoint_path = f"results/checkpoints/single_band_{band_name}.pt"

    device = get_device()

    print("=" * 60)
    print(f"TRAINING SINGLE-BAND MODEL: {band_name.upper()}")
    print("=" * 60)
    print("Device:", device)

    train_loader, val_loader, test_loader = get_single_band_dataloaders(
        band_index=band_index,
        batch_size=16,
    )

    model = SingleBandEEGFormer().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Train Acc: {train_acc:.4f} "
            f"Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f} "
            f"Val Acc: {val_acc:.4f} "
            f"Val F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best {band_name} model")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFinal Test Results")
    print("-" * 60)
    print(f"Band: {band_name}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()