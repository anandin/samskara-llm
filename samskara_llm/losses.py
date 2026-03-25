"""
SamskaraLoss — extended multi-task loss.

Extends compute_loss from train.py:217-235 with additional supervision signals
for Manas signals, Dharma scoring, and option selection. All 6 loss components
map directly to ATMAN training record fields.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LossConfig


class SamskaraLoss(nn.Module):

    def __init__(self, config: LossConfig):
        super().__init__()
        self.w = config

    def forward(self, model_out: dict, targets: dict) -> dict:
        """
        Compute 6-component loss.

        Args:
            model_out: dict from SamskaraLLM.forward()
            targets: dict with keys:
                - target_ids: [B, T] next-token targets
                - elevation_target: [B] binary (0 or 1)
                - outcome_score: [B] in [-1, 1]
                - manas_signal_targets: [B, 6] binary signal presence
                - manas_signal_intensity_targets: [B, 6] intensity values
                - dharma_targets: [B, n_options] target dharma scores
                - selected_option: [B] index of best option
        Returns:
            dict with total loss and per-component losses.
        """
        losses = {}

        # 1. Generation loss (next-token prediction)
        # model_out["logits"]: [B, T, V], target_ids: [B, T]
        logits = model_out["logits"]
        target_ids = targets["target_ids"]
        # Shift for autoregressive: predict next token from each position
        shift_logits = logits[:, :-1, :].contiguous()
        shift_targets = target_ids[:, 1:].contiguous()
        losses["generation"] = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_targets.view(-1),
            ignore_index=-100,
        )

        # 2. Elevation loss (router accuracy)
        losses["elevation"] = F.binary_cross_entropy(
            model_out["elevation_score"],
            targets["elevation_target"].float(),
        )

        # 3. Karma loss (memory-outcome alignment)
        # Reuse from train.py: MSE between chitta attention mean and normalized outcome
        losses["karma"] = F.mse_loss(
            model_out["chitta_attention"].mean(dim=-1),
            (targets["outcome_score"] + 1) / 2,
        )

        # 4. Manas signal loss (signal type classification + intensity)
        if "manas_signal_targets" in targets:
            losses["manas_signal"] = (
                F.binary_cross_entropy_with_logits(
                    model_out["manas_signal_logits"],
                    targets["manas_signal_targets"].float(),
                )
                + F.mse_loss(
                    model_out["manas_signal_intensities"],
                    targets["manas_signal_intensity_targets"],
                )
            ) / 2

        # 5. Dharma loss (ethical score accuracy)
        if "dharma_targets" in targets:
            losses["dharma"] = F.mse_loss(
                model_out["buddhi_dharma_scores"],
                targets["dharma_targets"],
            )

        # 6. Option selection loss (which option was chosen)
        if "selected_option" in targets:
            losses["option_selection"] = F.cross_entropy(
                model_out["buddhi_dharma_scores"],
                targets["selected_option"].long(),
            )

        # Weighted total
        total = (
            self.w.generation_weight * losses["generation"]
            + self.w.elevation_weight * losses["elevation"]
            + self.w.karma_weight * losses["karma"]
        )
        if "manas_signal" in losses:
            total = total + self.w.manas_signal_weight * losses["manas_signal"]
        if "dharma" in losses:
            total = total + self.w.dharma_weight * losses["dharma"]
        if "option_selection" in losses:
            total = total + self.w.option_selection_weight * losses["option_selection"]

        losses["total"] = total
        return losses
