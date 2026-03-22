# Samskara-LLM Autoresearch Program

## Goal

Minimize **val_bpb** (validation bits per byte) on a combined GSM8K + ETHICS dataset.
Lower val_bpb = better language modeling and cognitive architecture.

The model is a miniaturized Samskara architecture:
- **Chitta**: differentiable memory retrieval (seed bank → soft top-k)
- **Manas**: fast transformer processing (high temperature, associative)
- **ElevationRouter**: gate that blends Manas vs Buddhi outputs
- **Buddhi**: deliberate transformer processing (low temperature, focused)
- **lm_head**: vocabulary projection (weight-tied to embeddings)

## What You Can Modify in `train.py`

### Hyperparameters block (top of file — clearly marked)
- `n_seeds` — Chitta memory bank size (try 200–2000)
- `d_model` — embedding/hidden dimension (try 128, 256, 512; larger = slower)
- `manas_layers`, `buddhi_layers` — transformer depth per module
- `n_heads` — attention heads (must divide d_model evenly)
- `manas_temp`, `buddhi_temp` — attention temperatures
- `chitta_top_k` — retrieved seeds per query (try 10–50)
- `elevation_threshold` — gate cutoff for hard inference
- `loss_weights` — relative weighting of generation / elevation / karma losses
- `batch_size`, `lr`, `max_iters`, `max_seq_len`

### Architecture changes (inside the model classes)
- Optimizer choice (AdamW, SGD, RMSProp) and schedule
- Chitta query projection design
- Transformer feedforward multiplier (currently 4×)
- Activation function (GELU → SiLU, ReLU)
- Residual connections, layer norm placement
- Additional Chitta integration points (not just at the start)

## Constraints

- **Do NOT** modify the data loading block (below the hyperparameters section)
- **Do NOT** modify the evaluation block or the final `print(f"val_bpb: {val_bpb:.4f}")` line
- Training must complete within 5 minutes of wall clock time (the time gate enforces this)
- `n_heads` must divide `d_model` evenly

## Research Log

| Experiment | Key Changes | val_bpb | Notes |
|-----------|-------------|---------|-------|
| baseline  | default hyperparams | — | first run |

## Current Best

val_bpb: [TBD — fill in after first run]

## Hypothesis for Next Experiment

[Agent fills this in before each run]

## Tips

- Start with small changes; large architecture changes often break training
- If loss explodes, try lower lr or smaller d_model
- The elevation loss helps the router learn when to escalate — don't zero it out
- Chitta with too many seeds relative to d_model tends to underfit; try keeping n_seeds ≤ 4 × d_model
- Weight tying (lm_head.weight = embed.weight) saves parameters and usually helps
