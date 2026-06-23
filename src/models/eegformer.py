import torch
import torch.nn as nn


class BandEncoder(nn.Module):
    def __init__(self, num_channels=19, embed_dim=128):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(num_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.squeeze(-1)
        return x


class EEGFormer(nn.Module):
    """
    Multi-Band EEG Transformer for dementia classification.

    Input:
        (batch, 5, 19, 2500)

    Meaning:
        5    = Delta, Theta, Alpha, Beta, Gamma
        19   = EEG electrodes
        2500 = 5-second window samples

    Output:
        (batch, 3)
    """

    def __init__(
        self,
        num_bands=5,
        num_channels=19,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
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

        self.band_position_embedding = nn.Parameter(
            torch.randn(1, num_bands, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_band_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
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

        band_tokens = band_tokens + self.band_position_embedding

        band_tokens = self.cross_band_transformer(band_tokens)

        pooled = band_tokens.mean(dim=1)

        logits = self.classifier(pooled)

        return logits


if __name__ == "__main__":
    model = EEGFormer()

    dummy_input = torch.randn(16, 5, 19, 2500)

    output = model(dummy_input)

    print("=" * 60)
    print("EEGFORMER TEST")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)