import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.data.band_eeg_dataset import BandEEGDataset

train_dataset = BandEEGDataset("train")

counts = {0: 0, 1: 0, 2: 0}

for _, y in train_dataset:
    counts[int(y)] += 1

print("\nTrain Window Distribution")
print("-" * 40)
print(f"Alzheimer's:      {counts[0]}")
print(f"Healthy Control: {counts[1]}")
print(f"FTD:             {counts[2]}")
print(f"Total:           {sum(counts.values())}")