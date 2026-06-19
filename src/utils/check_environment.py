import torch
import mne
import numpy as np
import pandas as pd
import sklearn
import shap

print("=" * 50)
print("EEGFormer Environment Check")
print("=" * 50)

print(f"PyTorch Version: {torch.__version__}")
print(f"MNE Version: {mne.__version__}")
print(f"NumPy Version: {np.__version__}")
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-Learn Version: {sklearn.__version__}")
print(f"SHAP Version: {shap.__version__}")

print("\nPyTorch Backend")

if torch.backends.mps.is_available():
    print("Using Apple Silicon GPU (MPS)")
else:
    print("Using CPU")

print("\nEnvironment Ready")