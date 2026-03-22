"""Simple training script for Samskara-LLM."""
import torch, torch.nn as nn, json, os
from pathlib import Path

Path('outputs').mkdir(exist_ok=True)

with open('data/training/train.jsonl') as f:
    data = [json.loads(l) for l in f]

print(f"Loaded {len(data)} examples")

model = nn.Sequential(
    nn.Embedding(256, 512),
    nn.LSTM(512, 512, 2, batch_first=True)[0],
    nn.Linear(512, 256)
).cuda()

opt = torch.optim.Adam(model.parameters())

for epoch in range(3):
    total = 0
    for ex in data:
        ids = torch.tensor([ord(c) % 256 for c in ex['query'][:128]]).unsqueeze(0).cuda()
        out = model(ids)
        loss = nn.functional.cross_entropy(out.view(-1, 256), ids.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()
    print(f"Epoch {epoch+1}: {total/len(data):.4f}")
    torch.save(model.state_dict(), f'outputs/epoch{epoch+1}.pt')

torch.save(model.state_dict(), 'outputs/final.pt')
print("✅ TRAINING COMPLETE")
