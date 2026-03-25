"""
ManasLayer — fast reactive thinking.

Transformer layers running at high temperature (1.5). Associative, emotional, fast.
Fires six signal types: FEAR, DESIRE, PATTERN, RISK, OPPORTUNITY, NOISE.

Reuses TransformerBlock pattern from train.py:136-154 with parameterized dimensions.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..config import ManasConfig
from .transformer import TransformerBlock


@dataclass
class ManasOutput:
    hidden_states: torch.Tensor  # [B, T, D]
    pooled: torch.Tensor  # [B, D]
    signal_logits: torch.Tensor  # [B, n_signal_types]
    signal_intensities: torch.Tensor  # [B, n_signal_types]


class ManasLayer(nn.Module):

    def __init__(self, config: ManasConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                temperature=config.temperature,
            )
            for _ in range(config.n_layers)
        ])
        self.signal_head = nn.Linear(config.d_model, config.n_signal_types)
        self.intensity_head = nn.Linear(config.d_model, config.n_signal_types)

    def forward(self, x: torch.Tensor) -> ManasOutput:
        """
        Args:
            x: [B, T, D] input hidden states.
        Returns:
            ManasOutput with hidden_states, pooled, signal_logits, signal_intensities.
        """
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)  # [B, D]
        signal_logits = self.signal_head(pooled)  # [B, 6]
        signal_intensities = torch.sigmoid(self.intensity_head(pooled))  # [B, 6]
        return ManasOutput(
            hidden_states=x,
            pooled=pooled,
            signal_logits=signal_logits,
            signal_intensities=signal_intensities,
        )
