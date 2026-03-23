#!/usr/bin/env python3
"""
Samskara-LLM autoresearch training script.

This is the file the AI agent modifies. Humans should not edit it
directly during a research run — edit program.md instead.

The agent may modify ANYTHING in the HYPERPARAMETERS section below,
and may adjust the model architecture or optimizer. The data loading
block and the final `val_bpb:` print line must remain unchanged.

Usage:
    python autoresearch/train.py          # single experiment (~5 min)
    python autoresearch/run.py --hours 8  # overnight agent loop
"""

import math
import pickle
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── HYPERPARAMETERS (agent modifies this section) ────────────────────────────
n_seeds            = 500       # Chitta memory bank size
d_model            = 128       # Model dimension (keep small for 5-min runs)
manas_layers       = 1         # Manas transformer depth
buddhi_layers      = 3         # Buddhi transformer depth
n_heads            = 4         # Attention heads (d_model must be divisible)
manas_temp         = 1.5       # High temperature = associative/loose
buddhi_temp        = 0.3       # Low temperature = focused/grounded
chitta_top_k       = 4         # Seeds retrieved per query
elevation_threshold = 0.35     # Escalate to Buddhi above this gate value
n_dharma_rules     = 10        # Ethical constraint dimensions
n_options          = 2         # Buddhi generates N candidate options
synthesis_threshold = 0.5      # Trigger synthesis if divergence > this
loss_weights = {               # Multi-task loss weights
    "generation": 0.6,
    "elevation":  1.2,
    "karma":      1.5,
}
batch_size  = 16
lr          = 3e-4
max_iters   = 400              # Steps before time gate cuts off (~5 min)
max_seq_len = 128              # Truncate sequences to this length
val_batches = 40               # Batches used for val_bpb evaluation
# ─────────────────────────────────────────────────────────────────────────────


# ── DATA LOADING (do not modify) ─────────────────────────────────────────────

META_PATH = Path("data/autoresearch/meta.pkl")

if not META_PATH.exists():
    raise FileNotFoundError(
        f"{META_PATH} not found. Run: python autoresearch/prepare.py"
    )

with open(META_PATH, "rb") as f:
    meta = pickle.load(f)

vocab_size  = meta["vocab_size"]
train_raw   = meta["train"]
val_raw     = meta["val"]


class SamskaraDataset(Dataset):
    def __init__(self, records, max_len):
        self.records = records
        self.max_len = max_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        ids = rec["ids"][: self.max_len]
        return {
            "input_ids":        torch.tensor(ids[:-1] if len(ids) > 1 else ids, dtype=torch.long),
            "target_ids":       torch.tensor(ids[1]  if len(ids) > 1 else ids[0], dtype=torch.long),
            "target_elevation": torch.tensor(rec["elevation"], dtype=torch.float32),
            "target_outcome":   torch.tensor(rec["outcome"],   dtype=torch.float32),
        }


def collate(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["input_ids"]
        padded[i, : seq.size(0)] = seq
    return {
        "input_ids":        padded,
        "target_ids":       torch.stack([b["target_ids"]       for b in batch]),
        "target_elevation": torch.stack([b["target_elevation"] for b in batch]),
        "target_outcome":   torch.stack([b["target_outcome"]   for b in batch]),
    }


train_loader = DataLoader(
    SamskaraDataset(train_raw, max_seq_len),
    batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True,
)
val_loader = DataLoader(
    SamskaraDataset(val_raw, max_seq_len),
    batch_size=batch_size, shuffle=False, collate_fn=collate, drop_last=False,
)

# ─────────────────────────────────────────────────────────────────────────────


# ── MODEL ─────────────────────────────────────────────────────────────────────

class ChittaEncoder(nn.Module):
    """Differentiable memory retrieval."""
    def __init__(self):
        super().__init__()
        self.seeds    = nn.Parameter(torch.randn(n_seeds, d_model) * 0.02)
        self.query_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [batch, d_model] — mean-pooled input
        q = self.query_proj(x)                              # [batch, d_model]
        scores = q @ self.seeds.T / math.sqrt(d_model)     # [batch, n_seeds]
        top_scores, top_idx = scores.topk(chitta_top_k, dim=-1)
        attn = F.softmax(top_scores, dim=-1)                # [batch, k]
        seeds_k = self.seeds[top_idx]                       # [batch, k, d_model]
        field = (attn.unsqueeze(-1) * seeds_k).sum(dim=1)  # [batch, d_model]
        return field, attn


class TransformerBlock(nn.Module):
    def __init__(self, temp=1.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.temp = temp

    def forward(self, x):
        # Scale keys/queries by temperature
        x2 = self.norm1(x)
        attn_out, _ = self.attn(x2 / self.temp, x2 / self.temp, x2)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SamskaraMini(nn.Module):
    """
    Minimal Samskara architecture sized for 5-minute training runs.
    Chitta → embed → Manas → ElevationRouter → (optional Buddhi) → lm_head
    """
    def __init__(self):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.chitta  = ChittaEncoder()
        self.manas   = nn.Sequential(*[TransformerBlock(manas_temp) for _ in range(manas_layers)])
        self.buddhi  = nn.Sequential(*[TransformerBlock(buddhi_temp) for _ in range(buddhi_layers)])
        self.router  = nn.Linear(d_model, 1)   # elevation gate
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.embed.weight

    def forward(self, input_ids):
        x = self.embed(input_ids)                           # [B, T, D]
        pooled = x.mean(dim=1)                              # [B, D]

        # Chitta memory
        chitta_field, chitta_attn = self.chitta(pooled)    # [B, D], [B, k]

        # Manas (fast) — inject Chitta field
        x = x + chitta_field.unsqueeze(1)
        manas_out = self.manas(x)                           # [B, T, D]
        manas_pooled = manas_out.mean(dim=1)                # [B, D]

        # Elevation gate
        gate = torch.sigmoid(self.router(manas_pooled)).squeeze(-1)  # [B]

        # Buddhi (slow) — always run during training (soft gate)
        buddhi_out_seq = self.buddhi(manas_out)
        buddhi_pooled  = buddhi_out_seq.mean(dim=1)         # [B, D]

        # Soft blend: gate * buddhi + (1 - gate) * manas
        cognitive = gate.unsqueeze(-1) * buddhi_pooled + \
                    (1 - gate).unsqueeze(-1) * manas_pooled  # [B, D]

        logits = self.lm_head(cognitive)                    # [B, vocab_size]

        return {
            "logits":          logits,
            "elevation_gate":  gate,
            "chitta_attention": chitta_attn,
            "stability":       (1.0 - gate).unsqueeze(-1),   # proxy
        }


# ── TRAINING ──────────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = SamskaraMini().to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {n_params:,}")

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)

def compute_loss(out, batch):
    logits         = out["logits"]                  # [B, V]
    target_ids     = batch["target_ids"].to(device) # [B]
    target_elev    = batch["target_elevation"].to(device)
    target_outcome = batch["target_outcome"].to(device)

    gen_loss  = F.cross_entropy(logits, target_ids)
    elev_loss = F.binary_cross_entropy(out["elevation_gate"], target_elev)
    karma_loss = F.mse_loss(
        out["chitta_attention"].mean(dim=-1),
        (target_outcome + 1) / 2,
    )

    total = (
        loss_weights["generation"] * gen_loss +
        loss_weights["elevation"]  * elev_loss +
        loss_weights["karma"]      * karma_loss
    )
    return total, gen_loss


t0 = time.time()
step = 0
model.train()
train_iter = iter(train_loader)

print(f"\nTraining for up to {max_iters} steps (5-minute wall-clock cap)...")

for step in range(max_iters):
    # Wall-clock time gate
    if time.time() - t0 > 5 * 60:
        print(f"  [time gate hit at step {step}]")
        break

    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    out  = model({k: v.to(device) for k, v in batch.items() if k == "input_ids"}["input_ids"])
    loss, _ = compute_loss(out, batch)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    if step % 50 == 0:
        elapsed = time.time() - t0
        print(f"  step {step:4d}  loss={loss.item():.4f}  t={elapsed:.0f}s")

# ── EVALUATION (do not modify) ─────────────────────────────────────────────────
# Metric: 1 - S3_5dim (lower is better, consistent with autoresearch convention)
# S3 dimensions: elevation_acc, retrieval_prec, karma_corr, stability, efficiency
# This rewards Antahkarana-specific behavior, not just next-token prediction.

model.eval()
s3_accum  = dict(elevation_acc=0.0, retrieval_prec=0.0,
                 karma_corr=0.0, stability=0.0, efficiency=0.0)
n_batches = 0

with torch.no_grad():
    for batch in val_loader:
        if n_batches >= val_batches:
            break

        out     = model(batch["input_ids"].to(device))
        gate    = out["elevation_gate"]                    # [B]
        attn    = out["chitta_attention"]                  # [B, k]
        stab    = out["stability"]                         # [B]
        elev_t  = batch["target_elevation"].to(device)    # [B]
        outcome = batch["target_outcome"].to(device)      # [B]

        # 1. ElevationRouter accuracy (ATMAN Fidelity)
        elevation_acc  = ((gate > 0.5).float() == elev_t).float().mean()

        # 2. Chitta retrieval precision (peak attention weight)
        retrieval_prec = attn.max(dim=-1)[0].mean()

        # 3. Karma correlation (do seed weights predict outcome quality?)
        attn_mean = attn.mean(dim=-1)
        if attn_mean.std() > 1e-6 and outcome.std() > 1e-6:
            karma_corr = torch.corrcoef(
                torch.stack([attn_mean, outcome])
            )[0, 1].clamp(0.0, 1.0)
        else:
            karma_corr = torch.tensor(0.0, device=device)

        # 4. Cognitive Health (stability from ElevationRouter)
        stability_val = stab.mean()

        # 5. Efficiency (elevation rate near 20% — not always fast, not always slow)
        elev_rate  = (gate > 0.5).float().mean()
        efficiency = (1.0 - (elev_rate - 0.2).abs() * 2).clamp(0.0, 1.0)

        s3_accum["elevation_acc"]  += elevation_acc.item()
        s3_accum["retrieval_prec"] += retrieval_prec.item()
        s3_accum["karma_corr"]     += karma_corr.item()
        s3_accum["stability"]      += stability_val.item()
        s3_accum["efficiency"]     += efficiency.item()
        n_batches += 1

n   = max(n_batches, 1)
s3  = (
    0.25 * s3_accum["elevation_acc"]  / n +
    0.25 * s3_accum["retrieval_prec"] / n +
    0.20 * s3_accum["karma_corr"]     / n +
    0.15 * s3_accum["stability"]      / n +
    0.15 * s3_accum["efficiency"]     / n
)
val_metric = 1.0 - min(max(s3, 0.0), 1.0)   # lower = better S3 score

elapsed = time.time() - t0
print(f"\nDone in {elapsed:.0f}s  ({step + 1} steps)")
print(f"  S3 breakdown — "
      f"elev_acc={s3_accum['elevation_acc']/n:.3f}  "
      f"retr_prec={s3_accum['retrieval_prec']/n:.3f}  "
      f"karma_corr={s3_accum['karma_corr']/n:.3f}  "
      f"stability={s3_accum['stability']/n:.3f}  "
      f"efficiency={s3_accum['efficiency']/n:.3f}")
print(f"val_bpb: {val_metric:.4f}")
