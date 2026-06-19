from pathlib import Path
import mne

DATA_DIR = Path("data/raw/ds004504")

FREQ_BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}


def find_first_eeg_file():
    eeg_files = sorted(DATA_DIR.glob("sub-*/eeg/*.set"))

    if len(eeg_files) == 0:
        raise FileNotFoundError("No EEG .set files found.")

    return eeg_files[0]


def load_eeg_file(eeg_file):
    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=True,
        verbose=False
    )

    return raw


def filter_into_bands(raw):

    
    band_signals = {}

    for band_name, frequency_range in FREQ_BANDS.items():
        low_freq, high_freq = frequency_range

        print(f"Filtering {band_name}: {low_freq}-{high_freq} Hz")

        filtered_raw = raw.copy().filter(
            l_freq=low_freq,
            h_freq=high_freq,
            verbose=False
        )

        band_signals[band_name] = filtered_raw.get_data()

    return band_signals


def main():
    eeg_file = find_first_eeg_file()

    print("=" * 60)
    print("EEG FREQUENCY BAND FILTERING")
    print("=" * 60)

    print(f"\nLoaded EEG file: {eeg_file}")

    raw = load_eeg_file(eeg_file)

    original_data = raw.get_data()

    print(f"\nOriginal signal shape: {original_data.shape}")
    print(f"Sampling frequency: {raw.info['sfreq']} Hz")

    band_signals = filter_into_bands(raw)

    print("\nFiltered Band Shapes:")
    print("-" * 60)

    for band_name, signal in band_signals.items():
        print(f"{band_name}: {signal.shape}")


if __name__ == "__main__":
    main()




    """
    Decompose a raw EEG recording into canonical neural frequency bands.

    EEG signals are composed of multiple brain oscillations occurring
    simultaneously across different frequency ranges. Rather than treating
    the EEG as a single mixed signal, this function separates the recording
    into neuroscience-defined frequency bands:

        Delta : 0.5 - 4 Hz
        Theta : 4 - 8 Hz
        Alpha : 8 - 13 Hz
        Beta  : 13 - 30 Hz
        Gamma : 30 - 45 Hz

    Why this is important:
    ----------------------
    Alzheimer's disease and other dementias are known to alter specific
    frequency bands. Research has shown increased Delta/Theta activity and
    reduced Alpha activity in affected patients.

    By isolating each band, the model can learn band-specific neural
    characteristics instead of learning from a single mixed EEG signal.

    In the proposed EEGFormer architecture, each frequency band will be
    processed by a dedicated Transformer encoder before cross-band attention
    is applied.

    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG recording loaded using MNE.

    Returns
    -------
    dict
        Dictionary mapping band names to filtered EEG signals.

        Example:
        {
            "delta": ndarray (19, 299900),
            "theta": ndarray (19, 299900),
            "alpha": ndarray (19, 299900),
            "beta": ndarray (19, 299900),
            "gamma": ndarray (19, 299900)
        }

    Notes
    -----
    Filtering does not change signal dimensions.

    Example:

        Original EEG shape:
            (19, 299900)

        Delta shape:
            (19, 299900)

        Theta shape:
            (19, 299900)

    The number of channels and timestamps remain unchanged;
    only the frequency content of the signal is modified.
    """