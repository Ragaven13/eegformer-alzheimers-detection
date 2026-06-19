from pathlib import Path
import pandas as pd

DATA_DIR = Path("data/raw/ds004504")

def load_participants():
    participants_path = DATA_DIR / "participants.tsv"
    print("Looking for:", participants_path.resolve())

    if not participants_path.exists():
        raise FileNotFoundError(f"participants.tsv not found at {participants_path}")

    df = pd.read_csv(participants_path, sep="\t")
    return df

def list_subjects():
    return sorted([p.name for p in DATA_DIR.glob("sub-*") if p.is_dir()])

if __name__ == "__main__":
    participants = load_participants()
    subjects = list_subjects()

    print("Participants shape:", participants.shape)
    print("Columns:", participants.columns.tolist())
    print(participants.head())

    print("Number of subject folders:", len(subjects))
    print("First 10 subjects:", subjects[:10])
