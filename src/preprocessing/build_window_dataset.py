from pathlib import Path
import numpy as np
import pandas as pd
import mne

DATA_DIR = Path("data/raw/ds004504")
OUTPUT_DIR = Path("data/processed")

WINDOW_SECONDS = 5

LABEL_MAP = {
    "A": 0,  # Alzheimer's Disease
    "C": 1,  # Healthy Control
    "F": 2,  # Frontotemporal Dementia
}


def load_participants():
    participants_path = DATA_DIR / "participants.tsv"
    return pd.read_csv(participants_path, sep="\t")


def load_subject_eeg(subject_id):
    eeg_file = DATA_DIR / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose=False
    )

    return raw


def create_windows(signal, sfreq, window_seconds):
    window_size = int(sfreq * window_seconds)
    total_samples = signal.shape[1]

    windows = []

    for start in range(0, total_samples - window_size + 1, window_size):
        end = start + window_size
        window = signal[:, start:end]
        windows.append(window)

    return np.array(windows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    participants = load_participants()

    all_windows = []
    all_labels = []
    all_subject_ids = []

    print("=" * 60)
    print("BUILDING FULL EEG WINDOW DATASET")
    print("=" * 60)

    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        group = row["Group"]

        label = LABEL_MAP[group]

        print(f"\nProcessing {subject_id} | Group: {group} | Label: {label}")

        raw = load_subject_eeg(subject_id)

        signal = raw.get_data()
        sfreq = raw.info["sfreq"]

        windows = create_windows(
            signal=signal,
            sfreq=sfreq,
            window_seconds=WINDOW_SECONDS
        )

        labels = np.full(
            shape=(len(windows),),
            fill_value=label
        )

        subject_ids = np.full(
            shape=(len(windows),),
            fill_value=subject_id,
            dtype=object
        )

        all_windows.append(windows)
        all_labels.append(labels)
        all_subject_ids.append(subject_ids)

        print(f"Created windows: {windows.shape}")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0)

    print("\n" + "=" * 60)
    print("FINAL DATASET")
    print("=" * 60)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"subject_ids shape: {subject_ids.shape}")

    np.save(OUTPUT_DIR / "X_windows.npy", X)
    np.save(OUTPUT_DIR / "y_labels.npy", y)
    np.save(OUTPUT_DIR / "subject_ids.npy", subject_ids)

    print("\nSaved files:")
    print(OUTPUT_DIR / "X_windows.npy")
    print(OUTPUT_DIR / "y_labels.npy")
    print(OUTPUT_DIR / "subject_ids.npy")


if __name__ == "__main__":
    main()