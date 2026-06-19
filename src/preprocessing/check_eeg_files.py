from pathlib import Path

DATA_DIR = Path("data/raw/ds004504")


def main():
    eeg_dir = DATA_DIR / "sub-001" / "eeg"

    print("=" * 60)
    print("CHECKING EEG FILES")
    print("=" * 60)

    print("\nSub-001 EEG folder:")
    print(eeg_dir)

    print("\nFiles inside sub-001/eeg:")
    for file in sorted(eeg_dir.iterdir()):
        print(file.name)

    print("\nAll file extensions in dataset:")
    extensions = {}

    for file in DATA_DIR.glob("sub-*/eeg/*"):
        ext = file.suffix
        extensions[ext] = extensions.get(ext, 0) + 1

    for ext, count in sorted(extensions.items()):
        print(f"{ext}: {count}")


if __name__ == "__main__":
    main()