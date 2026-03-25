"""
ElevationRouter — measures post-dialogue cognitive shift.

NOT a binary on/off gate. Measures the degree of shift after the Manas-Buddhi
dialogue. High shift = Buddhi dominated (hard question). Low shift = Manas
dominated (easy question). This is the Vedantic Viveka process.

Evolves the simple router from train.py:168 into a shift-measurement mechanism.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ElevationConfig


class ElevationRouter(nn.Module):

    def __init__(self, config: ElevationConfig):
        super().__init__()
        d = config.d_model
        self.shift_proj = nn.Linear(d * 2, d)
        self.score_head = nn.Linear(d, 1)

    def forward(
        self, pre_dialogue: torch.Tensor, post_dialogue: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pre_dialogue: [B, D] pooled state before Manas-Buddhi dialogue.
            post_dialogue: [B, D] pooled state after dialogue.
        Returns:
            elevation_score: [B] in [0, 1] — degree of cognitive effort.
        """
        delta = post_dialogue - pre_dialogue  # [B, D]
        combined = torch.cat([delta, pre_dialogue], dim=-1)  # [B, 2D]
        h = F.gelu(self.shift_proj(combined))  # [B, D]
        return torch.sigmoid(self.score_head(h)).squeeze(-1)  # [B]
