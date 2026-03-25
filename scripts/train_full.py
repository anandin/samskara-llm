#!/usr/bin/env python3
"""
Full SamskaraLLM training script for Llama 3.1 8B + Antahkarana layers.

Loads ATMAN JSONL records, tokenizes with Llama tokenizer, trains with
6-component loss and per-layer learning rates.

Usage:
    python scripts/train_full.py --data-dir data/training --epochs 3
    python scripts/train_full.py --data-dir data/training --epochs 3 --bf16
    python scripts/train_full.py --dry-run  # verify data loading only

Env vars:
    HF_TOKEN — HuggingFace token for gated Llama model access
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from samskara_llm.config import SamskaraConfig
from samskara_llm.model import SamskaraLLM
from samskara_llm.losses import SamskaraLoss
from samskara_llm.metrics import S3Metric
from samskara_llm.lora import get_param_groups

SIGNAL_NAMES = ["FEAR", "DESIRE", "PATTERN", "RISK", "OPPORTUNITY", "NOISE"]


class ATMANDataset(Dataset):
    """Dataset built from ATMAN JSONL records."""

    def __init__(self, records: list, tokenizer, max_seq_len: int, n_options: int = 4):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.n_options = n_options

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]

        # Tokenize the linearized text
        text = rec.get("text", rec.get("scenario", ""))
        encoding = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].squeeze(0)  # [T]

        # Elevation target
        elevation_target = float(rec.get("elevation_target", 0))

        # Outcome score
        outcome_score = float(rec.get("outcome_score", 0.0))

        # Manas signal targets: [6] binary presence + [6] intensity
        signal_targets = torch.zeros(6)
        signal_intensities = torch.zeros(6)
        for sig in rec.get("manas_signals", []):
            sig_type = sig.get("type", "")
            if sig_type in SIGNAL_NAMES:
                sig_idx = SIGNAL_NAMES.index(sig_type)
                signal_targets[sig_idx] = 1.0
                signal_intensities[sig_idx] = float(sig.get("intensity", 0.5))

        # Dharma targets: [n_options] scores
        dharma_targets = torch.zeros(self.n_options)
        buddhi_options = rec.get("buddhi_options", [])
        for i, opt in enumerate(buddhi_options[:self.n_options]):
            dharma_targets[i] = float(opt.get("dharma_score", 0.5))

        # Selected option
        selected_option = min(int(rec.get("selected_option", 0)), self.n_options - 1)

        return {
            "input_ids": input_ids,
            "elevation_target": torch.tensor(elevation_target),
            "outcome_score": torch.tensor(outcome_score),
            "manas_signal_targets": signal_targets,
            "manas_signal_intensity_targets": signal_intensities,
            "dharma_targets": dharma_targets,
            "selected_option": torch.tensor(selected_option),
        }


def collate_atman(batch):
    """Pad input_ids to max length in batch."""
    max_len = max(b["input_ids"].size(0) for b in batch)
    padded_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, b in enumerate(batch):
        seq = b["input_ids"]
        padded_ids[i, :seq.size(0)] = seq

    return {
        "input_ids": padded_ids,
        "target_ids": padded_ids.clone(),  # autoregressive: target = input shifted
        "elevation_target": torch.stack([b["elevation_target"] for b in batch]),
        "outcome_score": torch.stack([b["outcome_score"] for b in batch]),
        "manas_signal_targets": torch.stack([b["manas_signal_targets"] for b in batch]),
        "manas_signal_intensity_targets": torch.stack([b["manas_signal_intensity_targets"] for b in batch]),
        "dharma_targets": torch.stack([b["dharma_targets"] for b in batch]),
        "selected_option": torch.stack([b["selected_option"] for b in batch]),
    }


def load_atman_records(path: Path) -> list:
    """Load ATMAN JSONL file."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    parser = argparse.ArgumentParser(description="Train SamskaraLLM")
    parser.add_argument("--data-dir", type=str, default="data/training")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--dry-run", action="store_true", help="Verify data loading only")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found.")
        print("Run: python scripts/generate_atman_data.py")
        sys.exit(1)

    # Load data
    print("Loading ATMAN training data...")
    train_records = load_atman_records(train_path)
    val_records = load_atman_records(val_path) if val_path.exists() else []
    print(f"  Train: {len(train_records)} records")
    print(f"  Val:   {len(val_records)} records")

    if args.dry_run:
        print("\nDRY RUN — data loading verified.")
        print(f"Sample record keys: {list(train_records[0].keys())}")
        return

    # Load tokenizer
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build datasets
    config = SamskaraConfig()
    config.llama_model_name = args.model_name
    config.max_seq_len = args.max_seq_len
    config.batch_size = args.batch_size

    train_dataset = ATMANDataset(train_records, tokenizer, args.max_seq_len)
    val_dataset = ATMANDataset(val_records, tokenizer, args.max_seq_len) if val_records else None

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_atman, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_atman,
    ) if val_dataset else None

    # Build model
    print(f"\nBuilding SamskaraLLM...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamskaraLLM(config)
    model.load_llama(args.model_name)
    model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total params: {total:,}")
    print(f"  Trainable:    {trainable:,}")

    # Optimizer with per-layer learning rates
    param_groups = get_param_groups(model, config.neuroplasticity)
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1)

    # Loss and metrics
    loss_fn = SamskaraLoss(config.loss)
    s3_metric = S3Metric()

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if args.bf16 else None
    autocast_ctx = torch.amp.autocast("cuda", dtype=torch.bfloat16) if args.bf16 else torch.amp.autocast("cuda", enabled=False)

    # Checkpoint dir
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation}")
    print(f"  Mixed precision: {'bf16' if args.bf16 else 'fp32'}")
    print()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move to device
            targets = {k: v.to(device) for k, v in batch.items()}
            input_ids = targets.pop("input_ids")

            with autocast_ctx:
                model_out = model(input_ids)
                losses = loss_fn(model_out, targets)
                loss = losses["total"] / args.gradient_accumulation

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if (batch_idx + 1) % args.gradient_accumulation == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            epoch_loss += losses["total"].item()

            if batch_idx % 10 == 0:
                elapsed = time.time() - t0
                print(
                    f"  Epoch {epoch+1} | batch {batch_idx}/{len(train_loader)} | "
                    f"loss={losses['total'].item():.4f} | t={elapsed:.0f}s"
                )

        avg_loss = epoch_loss / max(len(train_loader), 1)
        print(f"\n  Epoch {epoch+1} complete | avg_loss={avg_loss:.4f}")

        # Validation
        if val_loader:
            model.eval()
            s3_metric.reset()
            val_loss_total = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    targets = {k: v.to(device) for k, v in batch.items()}
                    input_ids = targets.pop("input_ids")
                    model_out = model(input_ids)
                    losses = loss_fn(model_out, targets)
                    val_loss_total += losses["total"].item()
                    s3_metric.update(model_out, targets)

            metrics = s3_metric.compute()
            print(f"  Val loss={val_loss_total/len(val_loader):.4f} | "
                  f"val_bpb={metrics['val_bpb']:.4f} | S3={metrics['s3']:.4f}")
            print(f"    elev_acc={metrics['elevation_acc']:.3f} "
                  f"retr_prec={metrics['retrieval_prec']:.3f} "
                  f"karma_corr={metrics['karma_corr']:.3f} "
                  f"stability={metrics['stability']:.3f} "
                  f"efficiency={metrics['efficiency']:.3f} "
                  f"dharma={metrics['dharma_alignment']:.3f}")

        # Checkpoint
        ckpt_path = ckpt_dir / f"samskara_epoch{epoch+1}.pt"
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": {
                k: v for k, v in model.state_dict().items()
                if "llama" not in k or "lora" in k
            },
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}\n")

    print("Training complete.")


if __name__ == "__main__":
    main()
