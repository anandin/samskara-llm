#!/usr/bin/env python3
"""
Train Samskara-LLM on REAL HotpotQA data.
This takes HOURS, not seconds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import requests
import os
from pathlib import Path
import time

Path('outputs').mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Download HotpotQA if not exists
if not Path('data/hotpot_train.json').exists():
    print("Downloading HotpotQA dataset (~30MB)...")
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    r = requests.get(url, stream=True)
    Path('data').mkdir(exist_ok=True)
    with open('data/hotpot_train.json', 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Downloaded.")

# Load and convert data
print("Loading HotpotQA...")
with open('data/hotpot_train.json') as f:
    hotpot = json.load(f)

print(f"Total examples: {len(hotpot)}")

class HotpotATMAN(Dataset):
    def __init__(self, data, max_len=512):
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        # Combine question + supporting facts + answer
        context = " ".join([sent for _, sent in ex.get('supporting_facts', [])[:5]])
        text = f"Question: {ex['question']} Context: {context} Answer: {ex['answer']}"
        ids = [ord(c) % 256 for c in text[:self.max_len]]
        return torch.tensor(ids, dtype=torch.long)

def collate(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded

# Use subset for faster training (full dataset = 90K examples = 10+ hours)
subset = hotpot[:10000]  # 10K examples = ~2-3 hours
print(f"Using subset: {len(subset)} examples")

dataset = HotpotATMAN(subset)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=collate)

# Bigger model
class SamskaraLLM(nn.Module):
    def __init__(self, vocab=256, embed=1024, hidden=1024):
        super().__init__()
        self.embed = nn.Embedding(vocab, embed)
        self.lstm = nn.LSTM(embed, hidden, num_layers=3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden, vocab)
        
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = SamskaraLLM().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

print(f"\n🚀 Training on REAL HotpotQA data...")
print(f"Model: 1024 dim, 3 layers")
print(f"Data: {len(subset)} examples, 512 tokens each")
print(f"Expected time: 2-3 hours\n")

start = time.time()

for epoch in range(3):
    model.train()
    epoch_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
    
    for batch in pbar:
        batch = batch.to(device)
        if batch.size(1) < 2:
            continue
            
        input_seq = batch[:, :-1]
        target_seq = batch[:, 1:]
        
        logits = model(input_seq)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_seq.reshape(-1),
            ignore_index=0
        )
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg = epoch_loss / len(dataloader)
    elapsed = (time.time() - start) / 3600
    print(f"\nEpoch {epoch+1}: loss={avg:.4f}, time={elapsed:.2f}h")
    torch.save(model.state_dict(), f'outputs/hotpot_epoch{epoch+1}.pt')

torch.save(model.state_dict(), 'outputs/hotpot_final.pt')
print(f"\n✅ DONE - Total time: {(time.time()-start)/3600:.2f} hours")
print(f"Model saved: outputs/hotpot_final.pt")
