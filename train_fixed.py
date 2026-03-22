#!/usr/bin/env python3
"""
PROPERLY WORKING training script for Samskara-LLM.
Tested to actually train (loss decreases, takes hours).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import os
from pathlib import Path
import time

# Setup
Path('outputs').mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
class ATMANDataset(Dataset):
    def __init__(self, file, max_len=128):
        with open(file) as f:
            self.data = [json.loads(l) for l in f]
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        text = ex['query']
        # Char-level encoding
        ids = [ord(c) % 256 for c in text[:self.max_len]]
        return torch.tensor(ids, dtype=torch.long)

def collate_fn(batch):
    """Pad sequences to same length."""
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded

# Load data
print("Loading dataset...")
dataset = ATMANDataset('data/training/train.jsonl')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
print(f"Dataset: {len(dataset)} examples, {len(dataloader)} batches")

# Model
class SamskaraModel(nn.Module):
    def __init__(self, vocab_size=256, embed_dim=512, hidden_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

model = SamskaraModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
print("\n🚀 Starting training...")
print("This will take ~3-4 hours. Leave it running.\n")

start_time = time.time()

for epoch in range(3):
    model.train()
    epoch_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
    for batch in pbar:
        batch = batch.to(device)
        
        # Forward: predict next character
        input_seq = batch[:, :-1]  # All except last
        target_seq = batch[:, 1:]  # All except first
        
        logits = model(input_seq)
        
        # Loss: next character prediction
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1),
            ignore_index=0
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = epoch_loss / num_batches
    print(f"\nEpoch {epoch+1} complete. Average loss: {avg_loss:.4f}")
    
    # Save checkpoint
    checkpoint_path = f'outputs/checkpoint_epoch{epoch+1}.pt'
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
    }, checkpoint_path)
    print(f"Saved: {checkpoint_path}")

# Save final
torch.save(model.state_dict(), 'outputs/final_model.pt')

elapsed = time.time() - start_time
print(f"\n✅ TRAINING COMPLETE")
print(f"Time: {elapsed/3600:.2f} hours")
print(f"Final model: outputs/final_model.pt")
print(f"\nYou can now use this model for inference!")
