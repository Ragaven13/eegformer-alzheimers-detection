import torch
import torch.nn as nn


class CrossBandAttention(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.attention_weights = None

    def forward(self, x):
        attn_output, attn_weights = self.attention(
            x,
            x,
            x,
            need_weights=True,
            average_attn_weights=False,
        )

        self.attention_weights = attn_weights

        x = self.norm1(x + attn_output)

        ffn_output = self.ffn(x)

        x = self.norm2(x + ffn_output)

        return x