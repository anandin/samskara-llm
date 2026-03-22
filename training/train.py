"""
Main training loop for Samskara-LLM.

Trains the Vedic cognitive architecture using multi-task objectives.
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml

from transformers import AutoTokenizer

from samskara_llm.model import create_samskara_llm
from training.data import load_datasets
from training.losses import samskara_loss, compute_s3_score


def load_config(config_path: str) -> dict:
    """Load training configuration."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
) -> dict:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_s3_scores = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        
        # Compute loss
        loss, loss_dict = samskara_loss(
            outputs,
            target_ids,
            target_signals=batch.get('target_signals'),
            target_elevation=batch.get('target_elevation'),
            target_dharma=batch.get('target_dharma'),
            target_outcome=batch.get('target_outcome'),
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        
        # Compute S3 score periodically
        if batch_idx % 10 == 0:
            s3_scores = compute_s3_score(outputs, batch)
            all_s3_scores.append(s3_scores)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'gen': f"{loss_dict['generation'].item():.4f}",
            's3': f"{s3_scores.get('s3_total', 0):.3f}",
        })
        
        # TensorBoard logging
        global_step = epoch * len(dataloader) + batch_idx
        writer.add_scalar('Loss/total', loss.item(), global_step)
        writer.add_scalar('Loss/generation', loss_dict['generation'].item(), global_step)
        writer.add_scalar('Loss/signal', loss_dict['signal'].item(), global_step)
        writer.add_scalar('Loss/elevation', loss_dict['elevation'].item(), global_step)
        writer.add_scalar('S3/total', s3_scores.get('s3_total', 0), global_step)
        
    avg_loss = total_loss / len(dataloader)
    avg_s3 = sum(s['s3_total'] for s in all_s3_scores) / len(all_s3_scores) if all_s3_scores else 0
    
    return {'loss': avg_loss, 's3_score': avg_s3}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate model."""
    model.eval()
    
    total_loss = 0.0
    all_s3_scores = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            
            outputs = model(input_ids)
            
            loss, _ = samskara_loss(outputs, target_ids)
            total_loss += loss.item()
            
            s3_scores = compute_s3_score(outputs, batch)
            all_s3_scores.append(s3_scores)
    
    avg_loss = total_loss / len(dataloader)
    avg_s3 = sum(s['s3_total'] for s in all_s3_scores) / len(all_s3_scores)
    
    return {'loss': avg_loss, 's3_score': avg_s3}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--config", default="configs/poc.yaml")
    parser.add_argument("--data-dir", default="data/training")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--wandb-project", default="samskara-llm")
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    config = load_config(args.config)
    print(f"Config: {config}")
    
    # Create model
    print("Creating model...")
    model = create_samskara_llm(args.model, **config['model'])
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )
    
    # TensorBoard
    writer = SummaryWriter(log_dir=f"{args.output_dir}/runs")
    
    # Load tokenizer and datasets
    print("Loading tokenizer and datasets...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    except Exception as e:
        print(f"Could not load tokenizer for {args.model}: {e}")
        print("Falling back to char-level encoding (no tokenizer).")
        tokenizer = None

    train_loader, val_loader = load_datasets(args, tokenizer)

    # Training loop
    print("\nStarting training...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, writer)
        print(f"Train — loss: {train_metrics['loss']:.4f}  S3: {train_metrics['s3_score']:.3f}")

        val_metrics = validate(model, val_loader, device)
        print(f"Val   — loss: {val_metrics['loss']:.4f}  S3: {val_metrics['s3_score']:.3f}")

        writer.add_scalar("Val/loss", val_metrics["loss"], epoch)
        writer.add_scalar("Val/s3", val_metrics["s3_score"], epoch)

        # Save per-epoch checkpoint
        ckpt_path = Path(args.output_dir) / f"checkpoint_epoch{epoch + 1}.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            },
            ckpt_path,
        )
        print(f"Checkpoint saved: {ckpt_path}")

    # Save model
    output_path = Path(args.output_dir) / "final_model.pt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    print(f"\n💾 Model saved to {output_path}")
    
    writer.close()


if __name__ == "__main__":
    main()
