from pathlib import Path
import numpy as np
import mne

DATA_DIR = Path("data/raw/ds004504")

WINDOW_SECONDS = 5


def load_first_eeg():
    eeg_files = sorted(DATA_DIR.glob("sub-*/eeg/*.set"))

    if len(eeg_files) == 0:
        raise FileNotFoundError("No EEG files found.")

    raw = mne.io.read_raw_eeglab(
        eeg_files[0],
        preload=True,
        verbose=False
    )

    return raw


def create_windows(signal, sfreq, window_seconds=5):
    """
    Split EEG signal into fixed-size windows.

    Parameters
    ----------
    signal : np.ndarray
        Shape: (channels, samples)

    sfreq : float
        Sampling frequency

    window_seconds : int
        Length of each window in seconds

    Returns
    -------
    np.ndarray
        Shape:
        (num_windows, channels, window_samples)
    """

    window_size = int(sfreq * window_seconds)

    total_samples = signal.shape[1]

    windows = []

    for start in range(
        0,
        total_samples - window_size + 1,
        window_size
    ):
        end = start + window_size

        window = signal[:, start:end]

        windows.append(window)

    return np.array(windows)


def main():
    raw = load_first_eeg()

    signal = raw.get_data()

    sfreq = raw.info["sfreq"]

    print("=" * 60)
    print("EEG WINDOW CREATION")
    print("=" * 60)

    print(f"\nOriginal Shape: {signal.shape}")
    print(f"Sampling Frequency: {sfreq}")

    windows = create_windows(
        signal,
        sfreq,
        window_seconds=WINDOW_SECONDS
    )

    print(f"\nWindow Length: {WINDOW_SECONDS} seconds")
    print(f"Window Samples: {int(sfreq * WINDOW_SECONDS)}")

    print("\nWindow Tensor Shape:")
    print(windows.shape)

    print("\nInterpretation:")
    print(f"{windows.shape[0]} windows")
    print(f"{windows.shape[1]} channels")
    print(f"{windows.shape[2]} samples per window")


if __name__ == "__main__":
    main()