"""
TransformerBlock — temperature-scaled attention.

Reused directly from train.py:136-154, parameterized for full model dimensions.
Temperature divides Q/K before attention (not logits after), which is more stable.
Pre-norm (LayerNorm before attention) + residuals for stable gradients.
"""

import torch.nn as nn


class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, n_heads: int, d_ff: int, temperature: float = 1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )
        self.temp = temperature

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2 / self.temp, x2 / self.temp, x2)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x
