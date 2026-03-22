#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import requests
from pathlib import Path
import time

Path('outputs').mkdir(exist_ok=True)
device = torch.device("cuda")
print(f"Device: {device}")

# Download
if not Path('data/hotpot_train.json').exists():
    print("Downloading HotpotQA...")
    url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
    r = requests.get(url)
    Path('data').mkdir(exist_ok=True)
    with open('data/hotpot_train.json', 'wb') as f:
        f.write(r.content)
    print("Done.")

# Load
with open('data/hotpot_train.json') as f:
    hotpot = json.load(f)

print(f"Loaded: {len(hotpot)} examples")

class HotpotDataset(Dataset):
    def __init__(self, data, max_len=512):
        self.data = data
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        ex = self.data[idx]
        # Just use question + context + answer
        context_parts = []
        for ctx in ex.get('context', [])[:3]:  # First 3 context docs
            if isinstance(ctx, list) and len(ctx) > 1:
                context_parts.extend(ctx[1][:3])  # First 3 sentences
        
        text = f"Q: {ex['question']} C: {' '.join(str(p) for p in context_parts)} A: {ex['answer']}"
        ids = [ord(c) % 256 for c in text[:self.max_len]]
        return torch.tensor(ids, dtype=torch.long)

def collate(batch):
    max_len = max(len(x) for x in batch)
    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(batch):
        padded[i, :len(seq)] = seq
    return padded

# Use subset
subset = hotpot[:5000]  # 5K examples = ~1-2 hours
dataset = HotpotDataset(subset)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate)

print(f"Training on {len(subset)} examples")

# Model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(256, 1024)
        self.lstm = nn.LSTM(1024, 1024, 3, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(1024, 256)
        
    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lstm(x)
        return self.fc(x)

model = Model().cuda()
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

print(f"\n🚀 Training...")
print(f"This will take 1-2 hours.\n")

start = time.time()

for epoch in range(3):
    model.train()
    total_loss = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/3")
    
    for batch in pbar:
        batch = batch.to(device)
        if batch.size(1) < 2:
            continue
            
        inp = batch[:, :-1]
        tgt = batch[:, 1:]
        
        out = model(inp)
        loss = F.cross_entropy(out.reshape(-1, 256), tgt.reshape(-1), ignore_index=0)
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg = total_loss / len(dataloader)
    elapsed = (time.time() - start) / 3600
    print(f"Epoch {epoch+1}: loss={avg:.4f}, time={elapsed:.2f}h")
    torch.save(model.state_dict(), f'outputs/epoch{epoch+1}.pt')

torch.save(model.state_dict(), 'outputs/final.pt')
print(f"\n✅ DONE - Time: {(time.time()-start)/3600:.2f} hours")
