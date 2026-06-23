from pathlib import Path
import numpy as np
import pandas as pd
import mne

DATA_DIR = Path("data/raw/ds004504")
OUTPUT_DIR = Path("data/processed/band_windows")

WINDOW_SECONDS = 5

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

LABEL_MAP = {
    "A": 0,
    "C": 1,
    "F": 2,
}


def create_windows(signal, sfreq, window_seconds):
    window_size = int(sfreq * window_seconds)
    total_samples = signal.shape[1]

    windows = []

    for start in range(0, total_samples - window_size + 1, window_size):
        end = start + window_size
        windows.append(signal[:, start:end])

    return np.array(windows, dtype=np.float32)


def load_subject_eeg(subject_id):
    eeg_file = DATA_DIR / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose=False,
    )

    return raw


def create_subject_band_windows(raw):
    sfreq = raw.info["sfreq"]
    subject_band_windows = []

    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
        print(f"  Filtering {band_name}: {low_freq}-{high_freq} Hz")

        filtered_raw = raw.copy().filter(
            l_freq=low_freq,
            h_freq=high_freq,
            verbose=False,
        )

        band_signal = filtered_raw.get_data()

        band_windows = create_windows(
            signal=band_signal,
            sfreq=sfreq,
            window_seconds=WINDOW_SECONDS,
        )

        subject_band_windows.append(band_windows)

    subject_band_windows = np.stack(subject_band_windows, axis=1)

    return subject_band_windows.astype(np.float32)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    participants = pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")

    print("=" * 60)
    print("BUILDING SUBJECT-WISE BAND WINDOW DATASET")
    print("=" * 60)

    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        group = row["Group"]
        label = LABEL_MAP[group]

        output_x = OUTPUT_DIR / f"{subject_id}_X_band.npy"
        output_y = OUTPUT_DIR / f"{subject_id}_y.npy"

        if output_x.exists() and output_y.exists():
            print(f"\nSkipping {subject_id}; already processed.")
            continue

        print(f"\nProcessing {subject_id} | Group: {group} | Label: {label}")

        raw = load_subject_eeg(subject_id)

        X_band = create_subject_band_windows(raw)

        y = np.full(
            shape=(X_band.shape[0],),
            fill_value=label,
            dtype=np.int64,
        )

        np.save(output_x, X_band)
        np.save(output_y, y)

        print(f"  Saved X: {output_x}")
        print(f"  Saved y: {output_y}")
        print(f"  Shape: {X_band.shape}")

    print("\nDone. Subject-wise band windows saved.")


if __name__ == "__main__":
    main()