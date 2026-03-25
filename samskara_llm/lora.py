"""
LoRA utilities — per-layer learning rate groups for the neuroplasticity gradient.

  Chitta karma/seeds:     lr=1e-3  (updates nightly)
  Manas layers:           lr=1e-5  (updates weekly)
  Buddhi layers:          lr=1e-6  (updates monthly)
  Ahamkara identity:      lr=0     (frozen, manual override only)
  Llama LoRA adapters:    lr=1e-5
  Cognitive infrastructure: lr=1e-4 (dharma, elevation, synthesis, dialogue)
"""

from .config import NeuroplasticityConfig


def get_param_groups(model, config: NeuroplasticityConfig) -> list:
    """
    Build optimizer parameter groups with per-layer learning rates.

    Args:
        model: SamskaraLLM instance.
        config: NeuroplasticityConfig with per-region LRs.
    Returns:
        List of dicts suitable for torch.optim.AdamW(param_groups).
    """
    groups = {
        "chitta": {"params": [], "lr": config.chitta_lr},
        "manas": {"params": [], "lr": config.manas_lr},
        "buddhi": {"params": [], "lr": config.buddhi_lr},
        "llama_lora": {"params": [], "lr": config.llama_lora_lr},
        "cognitive_infra": {"params": [], "lr": config.cognitive_infra_lr},
    }

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if "ahamkara" in name:
            # Ahamkara is frozen (lr=0), skip adding to optimizer
            if config.ahamkara_lr > 0:
                groups.setdefault("ahamkara", {"params": [], "lr": config.ahamkara_lr})
                groups["ahamkara"]["params"].append(param)
            continue

        if "chitta" in name:
            groups["chitta"]["params"].append(param)
        elif "manas" in name and "dialogue" not in name:
            groups["manas"]["params"].append(param)
        elif "buddhi" in name and "dialogue" not in name:
            groups["buddhi"]["params"].append(param)
        elif "lora" in name.lower():
            groups["llama_lora"]["params"].append(param)
        else:
            # dharma, elevation, synthesis, dialogue cross-projections
            groups["cognitive_infra"]["params"].append(param)

    # Filter out empty groups
    return [g for g in groups.values() if g["params"]]
