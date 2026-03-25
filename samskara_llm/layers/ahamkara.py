"""
AhamkaraLayer — organizational identity.

Context-conditioned identity layer. Has a frozen identity_prior (stable baseline
self) and a context_encoder that adapts to the current query. A blend gate
controls how much context shifts identity.

For enterprise: company values get loaded into identity_prior via load_identity().
Frozen during operation, manual override only.
"""

import torch
import torch.nn as nn

from ..config import AhamkaraConfig


class AhamkaraLayer(nn.Module):

    def __init__(self, config: AhamkaraConfig):
        super().__init__()
        d = config.d_model
        self.identity_prior = nn.Parameter(
            torch.zeros(d), requires_grad=False,
        )
        self.context_encoder = nn.Linear(d, d)
        self.blend_gate = nn.Sequential(
            nn.Linear(d * 2, d),
            nn.GELU(),
            nn.Linear(d, 1),
            nn.Sigmoid(),
        )

    def load_identity(self, path: str) -> None:
        """Load a company identity vector from a .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.shape != self.identity_prior.shape:
            raise ValueError(
                f"Identity shape mismatch: expected {self.identity_prior.shape}, "
                f"got {data.shape}"
            )
        self.identity_prior.data.copy_(data)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            context: [B, T, D] hidden states from Llama + previous layers.
        Returns:
            [B, T, D] context biased by organizational identity.
        """
        B = context.size(0)
        pooled = context.mean(dim=1)  # [B, D]
        ctx = self.context_encoder(pooled)  # [B, D]

        identity = self.identity_prior.unsqueeze(0).expand(B, -1)  # [B, D]
        gate_input = torch.cat([ctx, identity], dim=-1)  # [B, 2D]
        alpha = self.blend_gate(gate_input)  # [B, 1]

        blended = alpha * identity + (1 - alpha) * ctx  # [B, D]
        return context + blended.unsqueeze(1)  # [B, T, D]
