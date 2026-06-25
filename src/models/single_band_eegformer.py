import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.models.eegformer import BandEncoder


class SingleBandEEGFormer(nn.Module):

    def __init__(
        self,
        num_channels=19,
        embed_dim=128,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.encoder = BandEncoder(
            num_channels=num_channels,
            embed_dim=embed_dim,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):

        token = self.encoder(x)

        logits = self.classifier(token)

        return logits


if __name__ == "__main__":

    model = SingleBandEEGFormer()

    dummy_input = torch.randn(
        16,
        19,
        2500,
    )

    output = model(dummy_input)

    print("=" * 60)
    print("SINGLE BAND EEGFORMER TEST")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)