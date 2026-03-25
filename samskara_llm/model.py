"""
SamskaraLLM — the full Antahkarana architecture.

Wraps Llama 3.1 8B with cognitive layers:
  Ahamkara (identity) -> Chitta (memory) -> Manas-Buddhi dialogue (thinking)
  -> Dharma (ethics) -> Elevation (shift measurement) -> Synthesis (merge)
  -> Llama lm_head (token generation)

Forward pass flow matches the 10-step pipeline from the project spec.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SamskaraConfig
from .layers import (
    AhamkaraLayer,
    ChittaEncoder,
    ManasLayer,
    BuddhiLayer,
    DharmaLayer,
    ElevationRouter,
    SynthesisLayer,
)
from .dialogue import ManasBuddhiDialogue


class SamskaraLLM(nn.Module):

    def __init__(self, config: SamskaraConfig):
        super().__init__()
        self.config = config

        # Llama base model (loaded separately via from_pretrained)
        # We don't import transformers at module level to allow testing without it
        self.llama = None
        self.lm_head = None

        # Cognitive layers
        self.ahamkara = AhamkaraLayer(config.ahamkara)
        self.chitta = ChittaEncoder(config.chitta)
        self.manas = ManasLayer(config.manas)
        self.buddhi = BuddhiLayer(config.buddhi)
        self.dharma = DharmaLayer(config.dharma)
        self.elevation = ElevationRouter(config.elevation)
        self.synthesis = SynthesisLayer(config.synthesis)
        self.dialogue = ManasBuddhiDialogue(
            config.dialogue, d_model=config.chitta.d_model,
        )

    def load_llama(self, model_name: str = None):
        """Load Llama base model and apply PEFT LoRA."""
        from transformers import AutoModelForCausalLM
        from peft import LoraConfig, get_peft_model

        name = model_name or self.config.llama_model_name
        llama = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch.float16,
            device_map="auto",
        )

        # Apply LoRA to attention layers
        lora_config = LoraConfig(
            r=self.config.lora.llama_rank,
            lora_alpha=self.config.lora.llama_alpha,
            lora_dropout=self.config.lora.llama_dropout,
            target_modules=list(self.config.lora.target_modules),
            bias="none",
        )
        self.llama = get_peft_model(llama, lora_config)
        self.lm_head = llama.lm_head
        # Freeze base weights (LoRA adapters stay trainable)
        for name, param in self.llama.named_parameters():
            if "lora" not in name:
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor) -> dict:
        """
        Full Antahkarana forward pass.

        Args:
            input_ids: [B, T] token indices.
        Returns:
            Dict with logits and all intermediate cognitive signals.
        """
        if self.llama is None:
            raise RuntimeError(
                "Llama model not loaded. Call model.load_llama() first."
            )

        # Step 1: Llama hidden states
        llama_out = self.llama.model(input_ids)
        hidden = llama_out.last_hidden_state  # [B, T, D]

        # Step 2: Ahamkara identity bias
        hidden = self.ahamkara(hidden)  # [B, T, D]

        # Step 3: Chitta memory retrieval
        pooled = hidden.mean(dim=1)  # [B, D]
        chitta_field, chitta_attn = self.chitta(pooled)  # [B, D], [B, k]
        hidden = hidden + chitta_field.unsqueeze(1)  # inject memory

        # Step 4: Save pre-dialogue state for elevation measurement
        pre_dialogue = hidden.mean(dim=1)  # [B, D]

        # Step 5: 3-round Manas-Buddhi dialogue
        dialogue_out = self.dialogue(hidden, self.manas, self.buddhi)

        # Step 6: Dharma scoring on Buddhi options
        dharma_scores = self.dharma(dialogue_out.final_buddhi.options)  # [B, n_options]

        # Step 7: Option selection (differentiable soft selection)
        option_weights = F.softmax(dharma_scores, dim=-1)  # [B, n_options]
        buddhi_final = (
            option_weights.unsqueeze(-1) * dialogue_out.final_buddhi.options
        ).sum(dim=1)  # [B, D]

        # Step 8: Elevation measurement (post-dialogue shift)
        manas_final = dialogue_out.final_manas.pooled  # [B, D]
        post_dialogue = (manas_final + buddhi_final) / 2
        elevation_score = self.elevation(pre_dialogue, post_dialogue)  # [B]

        # Step 9: Synthesis (merge Manas and Buddhi)
        synth = self.synthesis(manas_final, buddhi_final)

        # Step 10: Generate logits via Llama's lm_head
        seq_output = hidden + synth.merged.unsqueeze(1)  # [B, T, D]
        logits = self.lm_head(seq_output)  # [B, T, vocab_size]

        return {
            "logits": logits,
            "elevation_score": elevation_score,
            "chitta_attention": chitta_attn,
            "manas_signal_logits": dialogue_out.final_manas.signal_logits,
            "manas_signal_intensities": dialogue_out.final_manas.signal_intensities,
            "buddhi_dharma_scores": dharma_scores,
            "divergence_score": synth.divergence_score,
            "stability": (1.0 - elevation_score).unsqueeze(-1),
        }
