# Samskara-LLM Autoresearch Program

## Goal

Minimize **val_bpb** (lower is better). This is `1 - S3_composite`, where S3 measures
how well the Antahkarana cognitive architecture is functioning — not just language modeling.

**val_bpb = 0.0** means perfect Antahkarana behavior.
**val_bpb = 1.0** means total failure on all cognitive dimensions.

## The 5 S3 Dimensions Being Optimized

| Dimension | Weight | What it measures | How to improve |
|-----------|--------|-----------------|----------------|
| **Elevation Accuracy** | 25% | ElevationRouter gate magnitude correlates with query complexity — high gate = Buddhi dominated the dialogue; low gate = Manas dominated | Tune `elevation_threshold`, `manas_temp`, `buddhi_temp` |
| **Retrieval Precision** | 25% | Chitta memory bank retrieves sharply (high peak attention weight) | Increase `n_seeds`, tune `chitta_top_k`, improve query projection |
| **Karma Correlation** | 20% | Seed attention weights correlate with actual outcome quality | Adjust `loss_weights["karma"]`, try different Chitta scoring heads |
| **Stability** | 15% | ElevationRouter's stability signal is consistently high | Lower `elevation_threshold`, regularize Manas signals |
| **Efficiency** | 15% | Elevation rate near 20% — system escalates selectively, not always/never | The router should escalate ~1 in 5 queries |

## What You Can Modify in `train.py`

### Hyperparameters block (top of file — clearly marked)
- `n_seeds` — Chitta memory bank size (try 200–2000)
- `d_model` — embedding/hidden dimension (try 128, 256, 512; larger = slower)
- `manas_layers`, `buddhi_layers` — transformer depth per module
- `n_heads` — attention heads (must divide d_model evenly)
- `manas_temp` — high temp = loose/associative Manas (try 1.0–2.5)
- `buddhi_temp` — low temp = focused Buddhi (try 0.1–0.5)
- `chitta_top_k` — retrieved seeds per query (try 10–50)
- `elevation_threshold` — gate cutoff (try 0.3–0.8; lower = escalate more)
- `loss_weights` — relative weighting of generation / elevation / karma losses
- `batch_size`, `lr`, `max_iters`, `max_seq_len`

### Architecture changes (inside the model classes)
- Optimizer choice (AdamW, SGD, RMSProp) and LR schedule
- Chitta query projection design (linear vs MLP)
- Transformer feedforward multiplier (currently 4×)
- Activation function (GELU → SiLU)
- How the Chitta field is injected into Manas (additive vs cross-attention)

## Constraints

- **Do NOT** modify the data loading block (below the hyperparameters section)
- **Do NOT** modify the evaluation block or the final `print(f"val_bpb: ...")` line
- Training must complete within 5 minutes of wall clock time
- `n_heads` must divide `d_model` evenly

## Failure Modes to Avoid

- **Elevation collapse**: router gate always near 0 or 1 → output is always pure Manas or pure Buddhi, losing the soft blend benefit; efficiency penalizes extremes
- **Attention spreading**: Chitta attends uniformly → retrieval_prec ≈ 1/k, not sharp
- **Karma noise**: if batch has no outcome variance, karma_corr defaults to 0 — vary the loss_weights["karma"] to help
- **Instability collapse**: if Manas noise is too high → stability crashes

## Research Log

| Experiment | Key Changes | val_bpb | S3 score | Notes |
|-----------|-------------|---------|----------|-------|
| baseline  | default hyperparams | — | — | first run |

## Current Best

val_bpb: [TBD — fill in after first run]

## Hypothesis for Next Experiment

[Agent fills this in before each run]

## Tips

- Elevation accuracy is 25% of the score — the router is the highest-leverage component; it now runs *after* the 3-round dialogue and tunes the Manas/Buddhi blend ratio (not whether to run the loop)
- Start by tuning `elevation_threshold` and `manas_temp` before touching architecture
- Chitta retrieval precision is limited by `chitta_top_k` — lower k forces sharper attention
- The karma loss trains Chitta to retrieve seeds that actually help → raise `loss_weights["karma"]`
- Efficiency penalizes both extremes: if elevation rate strays far from 20%, score drops fast
