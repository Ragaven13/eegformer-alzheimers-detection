import torch
import torch.nn as nn


class EEGLSTMBaseline(nn.Module):
    """
    LSTM baseline for EEG window classification.

    Input shape:
        (batch_size, 19, 2500)

    Internally converted to:
        (batch_size, 2500, 19)

    Meaning:
        2500 = time steps
        19 = EEG channel features per time step

    Output shape:
        (batch_size, 3)
    """

    def __init__(
        self,
        num_channels=19,
        hidden_size=64,
        num_layers=2,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes),
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)

        output, (hidden, cell) = self.lstm(x)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        final_hidden = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1,
        )

        logits = self.classifier(final_hidden)

        return logits


if __name__ == "__main__":
    model = EEGLSTMBaseline()

    dummy_input = torch.randn(32, 19, 2500)

    output = model(dummy_input)

    print("=" * 60)
    print("LSTM BASELINE TEST")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)