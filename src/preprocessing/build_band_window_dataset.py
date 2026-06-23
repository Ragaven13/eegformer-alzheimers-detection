from pathlib import Path
import numpy as np
import pandas as pd
import mne

DATA_DIR = Path("data/raw/ds004504")
OUTPUT_DIR = Path("data/processed")

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


def load_participants():
    return pd.read_csv(DATA_DIR / "participants.tsv", sep="\t")


def load_subject_eeg(subject_id):
    eeg_file = DATA_DIR / subject_id / "eeg" / f"{subject_id}_task-eyesclosed_eeg.set"

    if not eeg_file.exists():
        raise FileNotFoundError(f"EEG file not found: {eeg_file}")

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose=False,
    )

    return raw


def create_windows(signal, sfreq, window_seconds):
    window_size = int(sfreq * window_seconds)
    total_samples = signal.shape[1]

    windows = []

    for start in range(0, total_samples - window_size + 1, window_size):
        end = start + window_size
        windows.append(signal[:, start:end])

    return np.array(windows)


def filter_band(raw, low_freq, high_freq):
    filtered_raw = raw.copy().filter(
        l_freq=low_freq,
        h_freq=high_freq,
        verbose=False,
    )

    return filtered_raw.get_data()


def create_band_windows(raw):
    sfreq = raw.info["sfreq"]

    all_band_windows = []

    for band_name, (low_freq, high_freq) in FREQ_BANDS.items():
        print(f"  Filtering {band_name}: {low_freq}-{high_freq} Hz")

        band_signal = filter_band(raw, low_freq, high_freq)

        band_windows = create_windows(
            signal=band_signal,
            sfreq=sfreq,
            window_seconds=WINDOW_SECONDS,
        )

        all_band_windows.append(band_windows)

    band_windows = np.stack(all_band_windows, axis=1)

    return band_windows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    participants = load_participants()

    all_X = []
    all_y = []
    all_subject_ids = []

    print("=" * 60)
    print("BUILDING BAND-WINDOW EEG DATASET")
    print("=" * 60)

    for _, row in participants.iterrows():
        subject_id = row["participant_id"]
        group = row["Group"]
        label = LABEL_MAP[group]

        print(f"\nProcessing {subject_id} | Group: {group} | Label: {label}")

        raw = load_subject_eeg(subject_id)

        band_windows = create_band_windows(raw)

        labels = np.full(
            shape=(band_windows.shape[0],),
            fill_value=label,
        )

        subject_ids = np.full(
            shape=(band_windows.shape[0],),
            fill_value=subject_id,
            dtype=object,
        )

        all_X.append(band_windows)
        all_y.append(labels)
        all_subject_ids.append(subject_ids)

        print(f"  Band-window shape: {band_windows.shape}")

    X_band = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0)

    print("\n" + "=" * 60)
    print("FINAL BAND-WINDOW DATASET")
    print("=" * 60)
    print(f"X_band shape: {X_band.shape}")
    print(f"y shape: {y.shape}")
    print(f"subject_ids shape: {subject_ids.shape}")

    np.save(OUTPUT_DIR / "X_band_windows.npy", X_band)
    np.save(OUTPUT_DIR / "y_band_labels.npy", y)
    np.save(OUTPUT_DIR / "band_subject_ids.npy", subject_ids)

    print("\nSaved files:")
    print(OUTPUT_DIR / "X_band_windows.npy")
    print(OUTPUT_DIR / "y_band_labels.npy")
    print(OUTPUT_DIR / "band_subject_ids.npy")


if __name__ == "__main__":
    main()