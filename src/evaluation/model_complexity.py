import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch

from src.models.cnn_baseline import EEGCNNBaseline
from src.models.lstm_baseline import EEGLSTMBaseline
from src.models.transformer_baseline import EEGTransformerBaseline
from src.models.eegformer import EEGFormer


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model, dummy_input, device, runs=50):
    model.eval()
    model.to(device)
    dummy_input = dummy_input.to(device)

    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy_input)

        start = time.time()

        for _ in range(runs):
            _ = model(dummy_input)

        end = time.time()

    return (end - start) / runs


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main():
    device = get_device()

    models = {
        "CNN": (EEGCNNBaseline(), torch.randn(1, 19, 2500)),
        "LSTM": (EEGLSTMBaseline(), torch.randn(1, 19, 2500)),
        "Transformer": (EEGTransformerBaseline(), torch.randn(1, 19, 2500)),
        "EEGFormer": (EEGFormer(), torch.randn(1, 5, 19, 2500)),
    }

    print("=" * 60)
    print("MODEL COMPLEXITY ANALYSIS")
    print("=" * 60)
    print("Device:", device)

    for name, (model, dummy_input) in models.items():
        params = count_parameters(model)
        inference_time = measure_inference_time(model, dummy_input, device)

        print(f"\n{name}")
        print("-" * 40)
        print(f"Trainable parameters: {params:,}")
        print(f"Average inference time: {inference_time:.6f} seconds/sample")


if __name__ == "__main__":
    main()