"""
SamskaraLLM configuration dataclasses.

All hyperparameters for the Antahkarana architecture. Phase 1 validated values
(manas_temp=1.5, buddhi_temp=0.3, chitta_top_k=4, etc.) are preserved as defaults.
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class ChittaConfig:
    n_seeds: int = 10_000
    d_model: int = 4096
    top_k: int = 4  # validated in Phase 1


@dataclass
class ManasConfig:
    n_layers: int = 1  # validated in Phase 1
    d_model: int = 4096
    n_heads: int = 32
    d_ff: int = 14336
    temperature: float = 1.5  # validated in Phase 1
    n_signal_types: int = 6  # FEAR, DESIRE, PATTERN, RISK, OPPORTUNITY, NOISE
    signal_names: Tuple[str, ...] = (
        "FEAR", "DESIRE", "PATTERN", "RISK", "OPPORTUNITY", "NOISE",
    )


@dataclass
class BuddhiConfig:
    n_layers: int = 3  # validated in Phase 1
    d_model: int = 4096
    n_heads: int = 32
    d_ff: int = 14336
    temperature: float = 0.3  # validated in Phase 1
    n_options: int = 4


@dataclass
class AhamkaraConfig:
    d_model: int = 4096


@dataclass
class DharmaConfig:
    n_rules: int = 32
    d_model: int = 4096


@dataclass
class ElevationConfig:
    d_model: int = 4096


@dataclass
class SynthesisConfig:
    d_model: int = 4096


@dataclass
class DialogueConfig:
    n_rounds: int = 3


@dataclass
class LoRAConfig:
    llama_rank: int = 16
    llama_alpha: int = 32
    llama_dropout: float = 0.05
    target_modules: Tuple[str, ...] = ("q_proj", "v_proj")


@dataclass
class LossConfig:
    generation_weight: float = 0.6  # validated in Phase 1
    elevation_weight: float = 1.2  # validated in Phase 1
    karma_weight: float = 1.5  # validated in Phase 1
    manas_signal_weight: float = 1.0
    dharma_weight: float = 1.0
    option_selection_weight: float = 0.8


@dataclass
class NeuroplasticityConfig:
    """Per-layer learning rates for differentiated update cadence."""
    chitta_lr: float = 1e-3  # updates nightly
    manas_lr: float = 1e-5  # updates weekly
    buddhi_lr: float = 1e-6  # updates monthly
    ahamkara_lr: float = 0.0  # frozen, manual override only
    cognitive_infra_lr: float = 1e-4  # dharma, elevation, synthesis
    llama_lora_lr: float = 1e-5


@dataclass
class SamskaraConfig:
    llama_model_name: str = "meta-llama/Meta-Llama-3.1-8B"
    chitta: ChittaConfig = field(default_factory=ChittaConfig)
    manas: ManasConfig = field(default_factory=ManasConfig)
    buddhi: BuddhiConfig = field(default_factory=BuddhiConfig)
    ahamkara: AhamkaraConfig = field(default_factory=AhamkaraConfig)
    dharma: DharmaConfig = field(default_factory=DharmaConfig)
    elevation: ElevationConfig = field(default_factory=ElevationConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    dialogue: DialogueConfig = field(default_factory=DialogueConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    neuroplasticity: NeuroplasticityConfig = field(default_factory=NeuroplasticityConfig)
    max_seq_len: int = 2048
    batch_size: int = 4
