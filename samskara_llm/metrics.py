"""
S3 evaluation metric — extended from train.py:269-338.

6-dimensional composite metric rewarding Antahkarana-specific behavior:
  1. elevation_acc (20%)   — router correctly identifies hard vs easy
  2. retrieval_prec (20%)  — chitta retrieves sharply
  3. karma_corr (17%)      — memory predicts outcome quality
  4. stability (13%)       — cognitive health
  5. efficiency (15%)      — elevation rate near 20%
  6. dharma_alignment (15%) — selected options match dharma targets
"""

import torch


class S3Metric:

    def __init__(self):
        self.reset()

    def reset(self):
        self._accum = {
            "elevation_acc": 0.0,
            "retrieval_prec": 0.0,
            "karma_corr": 0.0,
            "stability": 0.0,
            "efficiency": 0.0,
            "dharma_alignment": 0.0,
        }
        self._n = 0

    @torch.no_grad()
    def update(self, model_out: dict, targets: dict):
        """Accumulate metrics for one batch."""
        gate = model_out["elevation_score"]
        attn = model_out["chitta_attention"]
        stab = model_out["stability"]
        elev_t = targets["elevation_target"]
        outcome = targets["outcome_score"]

        # 1. Elevation accuracy
        self._accum["elevation_acc"] += (
            ((gate > 0.5).float() == elev_t.float()).float().mean().item()
        )

        # 2. Retrieval precision (peak attention weight)
        self._accum["retrieval_prec"] += attn.max(dim=-1)[0].mean().item()

        # 3. Karma correlation
        attn_mean = attn.mean(dim=-1)
        if attn_mean.std() > 1e-6 and outcome.std() > 1e-6:
            corr = torch.corrcoef(
                torch.stack([attn_mean, outcome])
            )[0, 1].clamp(0.0, 1.0)
            self._accum["karma_corr"] += corr.item()

        # 4. Stability
        self._accum["stability"] += stab.mean().item()

        # 5. Efficiency (elevation rate near 20%)
        elev_rate = (gate > 0.5).float().mean()
        efficiency = (1.0 - (elev_rate - 0.2).abs() * 2).clamp(0.0, 1.0)
        self._accum["efficiency"] += efficiency.item()

        # 6. Dharma alignment
        if "dharma_targets" in targets and "buddhi_dharma_scores" in model_out:
            dharma_pred = model_out["buddhi_dharma_scores"]
            dharma_tgt = targets["dharma_targets"]
            alignment = 1.0 - (dharma_pred - dharma_tgt).abs().mean()
            self._accum["dharma_alignment"] += alignment.item()

        self._n += 1

    def compute(self) -> dict:
        """Compute final S3 score and per-dimension breakdown."""
        n = max(self._n, 1)
        dims = {k: v / n for k, v in self._accum.items()}

        s3 = (
            0.20 * dims["elevation_acc"]
            + 0.20 * dims["retrieval_prec"]
            + 0.17 * dims["karma_corr"]
            + 0.13 * dims["stability"]
            + 0.15 * dims["efficiency"]
            + 0.15 * dims["dharma_alignment"]
        )
        val_bpb = 1.0 - min(max(s3, 0.0), 1.0)

        return {
            "s3": s3,
            "val_bpb": val_bpb,
            **dims,
        }
