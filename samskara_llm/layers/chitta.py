"""
ChittaEncoder — organizational memory.

Differentiable memory retrieval over learned seed vectors. Extended from the
Phase 1 prototype (train.py:118-133) with:
  - Per-seed karma scores that weight retrieval
  - load_seeds / load_karma for enterprise memory seeding
  - save_seeds / save_karma for nightly batch persistence
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import ChittaConfig


class ChittaEncoder(nn.Module):

    def __init__(self, config: ChittaConfig):
        super().__init__()
        self.n_seeds = config.n_seeds
        self.d_model = config.d_model
        self.top_k = config.top_k

        self.seeds = nn.Parameter(
            torch.randn(config.n_seeds, config.d_model) * 0.02,
        )
        self.karma = nn.Parameter(torch.zeros(config.n_seeds))
        self.query_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def load_seeds(self, path: str) -> None:
        """Load pre-trained seed vectors from a .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.shape != self.seeds.shape:
            raise ValueError(
                f"Seeds shape mismatch: expected {self.seeds.shape}, got {data.shape}"
            )
        self.seeds.data.copy_(data)

    def save_seeds(self, path: str) -> None:
        torch.save(self.seeds.data.clone(), path)

    def load_karma(self, path: str) -> None:
        """Load per-seed karma scores from a .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=True)
        if data.shape != self.karma.shape:
            raise ValueError(
                f"Karma shape mismatch: expected {self.karma.shape}, got {data.shape}"
            )
        self.karma.data.copy_(data)

    def save_karma(self, path: str) -> None:
        torch.save(self.karma.data.clone(), path)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, D] mean-pooled hidden states.
        Returns:
            field: [B, D] retrieved memory field.
            attn: [B, top_k] attention weights over selected seeds.
        """
        q = self.query_proj(x)  # [B, D]
        scores = q @ self.seeds.T / math.sqrt(self.d_model)  # [B, n_seeds]

        # Karma-weighted retrieval: seeds with higher karma are easier to retrieve
        karma_weights = F.softmax(self.karma, dim=-1)  # [n_seeds]
        scores = scores + karma_weights.unsqueeze(0).log().clamp(min=-10)

        top_scores, top_idx = scores.topk(self.top_k, dim=-1)  # [B, k]
        attn = F.softmax(top_scores, dim=-1)  # [B, k]
        seeds_k = self.seeds[top_idx]  # [B, k, D]
        field = (attn.unsqueeze(-1) * seeds_k).sum(dim=1)  # [B, D]
        return field, attn
