"""
ManasBuddhiDialogue — 3-round iterative refinement.

Both Manas and Buddhi always run. Round 1: Manas processes, sends to Buddhi.
Round 2: Buddhi responds, sends back to Manas. Round 3: Manas updates, final
synthesis. Weights are shared across rounds (same modules called multiple times).

Fully differentiable — no detach, no hard stops.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from .config import DialogueConfig
from .layers.manas import ManasLayer, ManasOutput
from .layers.buddhi import BuddhiLayer, BuddhiOutput


@dataclass
class DialogueOutput:
    final_manas: ManasOutput
    final_buddhi: BuddhiOutput
    n_rounds: int


class ManasBuddhiDialogue(nn.Module):

    def __init__(self, config: DialogueConfig, d_model: int):
        super().__init__()
        self.n_rounds = config.n_rounds
        self.manas_to_buddhi = nn.Linear(d_model, d_model)
        self.buddhi_to_manas = nn.Linear(d_model, d_model)

    def forward(
        self,
        x: torch.Tensor,
        manas: ManasLayer,
        buddhi: BuddhiLayer,
    ) -> DialogueOutput:
        """
        3-round iterative dialogue between Manas and Buddhi.

        Args:
            x: [B, T, D] input hidden states (post-Chitta injection).
            manas: ManasLayer instance (shared weights across rounds).
            buddhi: BuddhiLayer instance (shared weights across rounds).
        Returns:
            DialogueOutput with final states from both paths.
        """
        manas_input = x
        buddhi_input = x
        manas_out = None
        buddhi_out = None

        for round_idx in range(self.n_rounds):
            # Manas processes
            manas_out = manas(manas_input)

            # Send Manas signal to Buddhi
            manas_signal = self.manas_to_buddhi(manas_out.pooled)  # [B, D]
            buddhi_input = buddhi_input + manas_signal.unsqueeze(1)  # [B, T, D]

            # Buddhi processes
            buddhi_out = buddhi(buddhi_input)

            # Send Buddhi signal back to Manas (except last round)
            if round_idx < self.n_rounds - 1:
                buddhi_signal = self.buddhi_to_manas(buddhi_out.pooled)  # [B, D]
                manas_input = manas_input + buddhi_signal.unsqueeze(1)  # [B, T, D]

        return DialogueOutput(
            final_manas=manas_out,
            final_buddhi=buddhi_out,
            n_rounds=self.n_rounds,
        )
