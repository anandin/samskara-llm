"""
BuddhiLayer — slow deliberate reasoning.

Transformer layers at low temperature (0.3). Generates multiple candidate option
representations that get scored by the DharmaLayer.

Reuses TransformerBlock pattern from train.py:136-154.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..config import BuddhiConfig
from .transformer import TransformerBlock


@dataclass
class BuddhiOutput:
    hidden_states: torch.Tensor  # [B, T, D]
    pooled: torch.Tensor  # [B, D]
    options: torch.Tensor  # [B, n_options, D]


class BuddhiLayer(nn.Module):

    def __init__(self, config: BuddhiConfig):
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
        self.option_projectors = nn.ModuleList([
            nn.Linear(config.d_model, config.d_model)
            for _ in range(config.n_options)
        ])

    def forward(self, x: torch.Tensor) -> BuddhiOutput:
        """
        Args:
            x: [B, T, D] input hidden states.
        Returns:
            BuddhiOutput with hidden_states, pooled, and options.
        """
        for layer in self.layers:
            x = layer(x)
        pooled = x.mean(dim=1)  # [B, D]
        options = torch.stack(
            [proj(pooled) for proj in self.option_projectors], dim=1,
        )  # [B, n_options, D]
        return BuddhiOutput(
            hidden_states=x,
            pooled=pooled,
            options=options,
        )
