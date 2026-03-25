"""
DharmaLayer — learnable ethical constraint embeddings.

Each Buddhi option gets scored against these constraints. Options with low
dharma compliance get penalized during selection.
"""

import math

import torch
import torch.nn as nn


from ..config import DharmaConfig


class DharmaLayer(nn.Module):

    def __init__(self, config: DharmaConfig):
        super().__init__()
        self.d_model = config.d_model
        self.constraint_embeddings = nn.Parameter(
            torch.randn(config.n_rules, config.d_model) * 0.02,
        )

    def forward(self, options: torch.Tensor) -> torch.Tensor:
        """
        Args:
            options: [B, n_options, D] candidate representations from Buddhi.
        Returns:
            dharma_scores: [B, n_options] in [0, 1] — ethical alignment per option.
        """
        # [B, n_options, D] @ [D, n_rules] -> [B, n_options, n_rules]
        alignment = options @ self.constraint_embeddings.T / math.sqrt(self.d_model)
        dharma_scores = torch.sigmoid(alignment.mean(dim=-1))  # [B, n_options]
        return dharma_scores
