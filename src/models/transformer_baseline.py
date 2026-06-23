import torch
import torch.nn as nn


class EEGTransformerBaseline(nn.Module):
    """
    Standard Transformer baseline for EEG window classification.

    Input shape:
        (batch_size, 19, 2500)

    Patch idea:
        2500 time samples are split into 50 patches.
        Each patch contains 50 time samples from all 19 EEG channels.

    Output shape:
        (batch_size, 3)
    """

    def __init__(
        self,
        num_channels=19,
        num_classes=3,
        window_samples=2500,
        patch_size=50,
        embed_dim=128,
        num_heads=4,
        num_layers=2,
        dropout=0.3,
    ):
        super().__init__()

        assert window_samples % patch_size == 0

        self.num_patches = window_samples // patch_size
        self.patch_dim = num_channels * patch_size

        self.patch_embedding = nn.Linear(
            self.patch_dim,
            embed_dim,
        )

        self.cls_token = nn.Parameter(
            torch.randn(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches + 1, embed_dim)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            batch_first=True,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = x.unfold(
            dimension=2,
            size=50,
            step=50,
        )

        x = x.permute(0, 2, 1, 3)

        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_dim,
        )

        x = self.patch_embedding(x)

        cls_tokens = self.cls_token.expand(
            batch_size,
            -1,
            -1,
        )

        x = torch.cat([cls_tokens, x], dim=1)

        x = x + self.position_embedding

        x = self.transformer_encoder(x)

        cls_output = x[:, 0, :]

        logits = self.classifier(cls_output)

        return logits


if __name__ == "__main__":
    model = EEGTransformerBaseline()

    dummy_input = torch.randn(32, 19, 2500)

    output = model(dummy_input)

    print("=" * 60)
    print("TRANSFORMER BASELINE TEST")
    print("=" * 60)
    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)