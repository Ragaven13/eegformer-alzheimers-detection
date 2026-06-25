import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn

from src.models.eegformer import BandEncoder


class EEGFormerNoCrossAttention(nn.Module):
    """
    EEGFormer ablation model without cross-band attention.
    """

    def __init__(
        self,
        num_bands=5,
        num_channels=19,
        embed_dim=128,
        num_classes=3,
        dropout=0.3,
    ):
        super().__init__()

        self.num_bands = num_bands

        self.band_encoders = nn.ModuleList(
            [
                BandEncoder(
                    num_channels=num_channels,
                    embed_dim=embed_dim,
                )
                for _ in range(num_bands)
            ]
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        band_tokens = []

        for band_index in range(self.num_bands):
            band_signal = x[:, band_index, :, :]
            band_token = self.band_encoders[band_index](band_signal)
            band_tokens.append(band_token)

        band_tokens = torch.stack(band_tokens, dim=1)

        pooled = band_tokens.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


if __name__ == "__main__":
    model = EEGFormerNoCrossAttention()

    dummy_input = torch.randn(16, 5, 19, 2500)

    output = model(dummy_input)

    print("=" * 60)
    print("EEGFORMER NO CROSS-ATTENTION TEST")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)