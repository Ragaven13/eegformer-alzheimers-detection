import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.metrics import accuracy_score, f1_score

from src.data.dataloader import get_dataloaders
from src.models.transformer_baseline import EEGTransformerBaseline


EPOCHS = 10
LEARNING_RATE = 1e-4
CHECKPOINT_PATH = "results/checkpoints/transformer_baseline.pt"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0
    all_preds = []
    all_labels = []

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(train_loader), accuracy, f1


def evaluate(model, data_loader, criterion, device):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    return total_loss / len(data_loader), accuracy, f1


def main():
    device = get_device()

    print("=" * 60)
    print("TRAINING TRANSFORMER BASELINE")
    print("=" * 60)
    print("Device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    model = EEGTransformerBaseline().to(device)

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
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print("Saved best Transformer model")

    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))

    test_loss, test_acc, test_f1 = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\nFinal Test Results")
    print("-" * 60)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1:.4f}")


if __name__ == "__main__":
    main()