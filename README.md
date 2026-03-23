# Samskara-LLM Training Infrastructure

**Vedic Cognitive Architecture as Neural Network**

This repo implements the Antahkarana (Chitta-Manus-Buddhi-Synthesis) directly in model weights, trained via the autoresearch pattern.

## Architecture Overview

```
Input Tokens
    ↓
[Chitta Encoder] → Differentiable retrieval from learned memory bank
    ↓
[Ahamkara] → Context-conditioned identity vector
    ↓
╔══ Iterative Dialogue (N=3 rounds) ════════════════════════════╗
║  [Manas Layers] ← Chitta + Ahamkara + Dharma (every round)   ║
║      ↕  (Buddhi's output feeds back into Manas each round)   ║
║  [Buddhi Layers] → Option generation scored by Dharma         ║
╚═══════════════════════════════════════════════════════════════╝
    ↓
[Elevation Router] → Measures degree of rational override (soft 0–1)
                     High gate = Buddhi dominated; Low gate = Manas dominated
    ↓
[Soft Blend] → elevation_gate × Buddhi + (1−gate) × Manas
    ↓
[Synthesis Layer] → Resolves final Manas/Buddhi divergence
    ↓
Output Tokens
```

## Quick Start

```bash
# 1. Setup environment
pip install -r requirements.txt

# 2. Generate training data from Samskara logs
python scripts/generate_atman_data.py --input samskara-logs/ --output data/training/

# 3. Train on single GPU (proof-of-concept)
python train.py --model qwen2.5-1.5b --config configs/poc.yaml

# 4. Run autoresearch (architecture search)
python autoresearch.py --base-config configs/poc.yaml --budget 10
```

## Cloud Training

For full-scale training (7B+ models):

```bash
# Deploy training harness to RunPod
python scripts/deploy_to_runpod.py --gpu A100 --model llama-3.1-8b

# Or use Lambda Labs
python scripts/deploy_to_lambda.py --gpu A10 --model qwen2.5-7b

# Monitor from Fly.io dashboard
fly logs --app samskara-llm-training
```

## Repository Structure

```
samskara-llm/
├── samskara_llm/           # Core model implementation
│   ├── model.py            # SamskaraLLM architecture
│   ├── chitta.py           # Differentiable memory encoder
│   ├── manas.py            # Reactive layers
│   ├── buddhi.py           # Discriminating layers
│   ├── router.py           # Elevation gating
│   ├── synthesis.py        # Conflict resolution
│   └── dharma.py           # Constitutional constraints
├── training/
│   ├── train.py            # Main training loop
│   ├── data.py             # ATMAN dataset loader
│   ├── losses.py           # Multi-task objectives
│   └── autoresearch.py     # Architecture optimization
├── configs/                # Model configurations
│   ├── poc.yaml           # Proof-of-concept (1.5B)
│   ├── small.yaml         # 3B model
│   └── production.yaml    # 8B model
├── scripts/               # Utilities
│   ├── generate_atman_data.py
│   ├── deploy_to_runpod.py
│   ├── deploy_to_lambda.py
│   └── benchmark.py
└── infrastructure/        # Cloud deployment
    ├── docker/
    ├── kubernetes/
    └── fly.io/
```

## Model Configurations

### Proof-of-Concept (1.5B params)
- Base: Qwen2.5-1.5B-Instruct
- Chitta: 10K seed memory
- Manas: 4 layers, temp=1.5
- Buddhi: 8 layers, temp=0.3
- Training: ~4 hours on A100

### Production (8B params)
- Base: Llama-3.1-8B or Qwen2.5-7B
- Chitta: 100K seed memory
- Manas: 8 layers, temp=1.5
- Buddhi: 16 layers, temp=0.3
- Training: ~2 days on 4x A100

## Training Data Format

```json
{
  "query": "Should we delay the product launch?",
  "domain": "strategy",
  "complexity": "high",
  "chitta_seeds": [
    {"content": "Past delayed launches succeeded", "vritti": "smriti", "karma": 0.8},
    {"content": "Market window closing", "vritti": "vikalpa", "karma": 0.3}
  ],
  "manas_output": {
    "signals": [
      {"tag": "FEAR", "content": "Missing market opportunity"},
      {"tag": "PATTERN", "content": "Past delays worked"}
    ]
  },
  "elevation_triggered": true,
  "buddhi_output": {
    "options": [
      {"text": "Launch on time with reduced scope", "dharma_score": 0.9},
      {"text": "Delay 2 weeks for polish", "dharma_score": 0.7}
    ],
    "recommendation": "Launch on time"
  },
  "synthesis_triggered": false,
  "final_response": "Based on pattern analysis, launching on time...",
  "outcome_quality": 1.0
}
```

## Evaluation Metrics (S3 Score)

| Dimension | Metric | Target |
|-----------|--------|--------|
| ATMAN Fidelity | `elevation_accuracy` — gate magnitude correlates with query complexity | >85% |
| Chitta Quality | `seed_retrieval_precision` | >75% |
| Divergence | `signal_preservation` | >90% |
| Karma | `outcome_prediction_accuracy` | >65% |
| Dharma | `compliance_rate` | >95% |
| Cognitive Health | `stability_score` | >0.7 |
| Efficiency | `tokens_per_decision` | <500 |

## License

MIT
