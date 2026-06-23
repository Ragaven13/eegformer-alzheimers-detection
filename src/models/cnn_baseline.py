import torch
import torch.nn as nn


class EEGCNNBaseline(nn.Module):
    """
    1D CNN baseline for EEG window classification.

    Input shape:
        (batch_size, 19, 2500)

    Meaning:
        batch_size = number of EEG windows
        19 = EEG channels/electrodes
        2500 = time samples per 5-second window

    Output shape:
        (batch_size, 3)

    Classes:
        0 = Alzheimer's
        1 = Healthy Control
        2 = Frontotemporal Dementia
    """

    def __init__(self, num_channels=19, num_classes=3):
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(
                in_channels=num_channels,
                out_channels=32,
                kernel_size=7,
                padding=3,
            ),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=5,
                padding=2,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits


if __name__ == "__main__":
    model = EEGCNNBaseline()

    dummy_input = torch.randn(32, 19, 2500)

    output = model(dummy_input)

    print("=" * 60)
    print("CNN BASELINE TEST")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)