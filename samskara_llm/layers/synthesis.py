"""
SynthesisLayer — merges final Manas and Buddhi states.

Computes divergence between the two cognitive paths. When divergence is high
(Manas and Buddhi strongly disagree), uses a learned blend. When low, simple
average. Replaces the soft blend from train.py:193-194 with a divergence-aware
mechanism.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


from ..config import SynthesisConfig


@dataclass
class SynthesisOutput:
    merged: torch.Tensor  # [B, D]
    divergence_score: torch.Tensor  # [B]


class SynthesisLayer(nn.Module):

    def __init__(self, config: SynthesisConfig):
        super().__init__()
        d = config.d_model
        self.divergence_net = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )
        self.blend_proj = nn.Linear(d * 2, d)

    def forward(
        self, manas_state: torch.Tensor, buddhi_state: torch.Tensor,
    ) -> SynthesisOutput:
        """
        Args:
            manas_state: [B, D] final Manas pooled representation.
            buddhi_state: [B, D] final Buddhi selected representation.
        Returns:
            SynthesisOutput with merged [B, D] and divergence_score [B].
        """
        combined = torch.cat([manas_state, buddhi_state], dim=-1)  # [B, 2D]
        divergence = self.divergence_net(combined)  # [B, 1]

        blended = self.blend_proj(combined)  # [B, D]
        simple_avg = (manas_state + buddhi_state) / 2  # [B, D]

        merged = divergence * blended + (1 - divergence) * simple_avg  # [B, D]
        return SynthesisOutput(
            merged=merged,
            divergence_score=divergence.squeeze(-1),
        )
