from pathlib import Path
import mne

DATA_DIR = Path("data/raw/ds004504")


def find_first_set_file():
    set_files = sorted(DATA_DIR.glob("sub-*/eeg/*.set"))

    if len(set_files) == 0:
        raise FileNotFoundError("No .set EEG files found.")

    return set_files[0]


def main():
    eeg_file = find_first_set_file()

    print("=" * 60)
    print("LOADING SINGLE EEG FILE")
    print("=" * 60)

    print(f"\nEEG file: {eeg_file}")

    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose=False
    )

    print("\nEEG Info")
    print("-" * 60)
    print(f"Number of channels: {raw.info['nchan']}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")
    print(f"Number of samples: {raw.n_times}")
    print(f"Duration: {raw.n_times / raw.info['sfreq']:.2f} seconds")

    print("\nChannel names:")
    print(raw.ch_names)

    data = raw.get_data()
    print("\nSignal shape:")
    print(data.shape)
    print("Shape meaning: (channels, time_samples)")


if __name__ == "__main__":
    main()