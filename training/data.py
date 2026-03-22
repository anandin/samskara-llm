"""
Data loading for Samskara-LLM training.

Supports two data sources:
  - ATMANDataset: loads from data/training/train.jsonl (full Vedic annotation format)
  - HotpotQADataset: falls back to HotpotQA multi-hop QA (downloaded automatically)
"""

import json
import random
from pathlib import Path

import requests
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode(text: str, tokenizer, max_len: int) -> list[int]:
    """Tokenize text; fall back to char-level if tokenizer unavailable."""
    if tokenizer is not None:
        ids = tokenizer.encode(text, add_special_tokens=False)
        return ids[:max_len]
    # Char-level fallback (matches train_fixed.py behaviour)
    return [ord(c) % tokenizer.vocab_size if tokenizer else ord(c) % 256
            for c in text[:max_len]]


def _char_encode(text: str, max_len: int) -> list[int]:
    return [ord(c) % 256 for c in text[:max_len]]


# ---------------------------------------------------------------------------
# ATMANDataset
# ---------------------------------------------------------------------------

class ATMANDataset(Dataset):
    """
    Loads the full ATMAN annotation format from a JSONL file.

    Expected record fields (all optional except `query`):
        query           : str   – input text
        final_response  : str   – target response (first token used as target_id)
        elevation_triggered : bool – did Buddhi activate?
        outcome_quality : float – [-1, 1] quality signal
    """

    def __init__(self, file_path: str, tokenizer=None, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ATMAN data file not found: {file_path}")
        with open(path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # Input: query text
        query = ex.get("query", "")
        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(query, add_special_tokens=False)[:self.max_len]
        else:
            input_ids = _char_encode(query, self.max_len)

        # Target: first token of the final response
        response = ex.get("final_response", query)
        if self.tokenizer is not None:
            resp_ids = self.tokenizer.encode(response, add_special_tokens=False)
            target_id = resp_ids[0] if resp_ids else 0
        else:
            target_id = _char_encode(response, 1)[0] if response else 0

        item = {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_id, dtype=torch.long),
        }

        # Optional supervision targets
        if "elevation_triggered" in ex:
            item["target_elevation"] = torch.tensor(
                float(ex["elevation_triggered"]), dtype=torch.float32
            )
        if "outcome_quality" in ex:
            item["target_outcome"] = torch.tensor(
                float(ex["outcome_quality"]), dtype=torch.float32
            )

        return item


# ---------------------------------------------------------------------------
# HotpotQADataset
# ---------------------------------------------------------------------------

HOTPOT_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
HOTPOT_PATH = Path("data/hotpot_train.json")


def _download_hotpot():
    HOTPOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading HotpotQA from {HOTPOT_URL} ...")
    r = requests.get(HOTPOT_URL, timeout=120)
    r.raise_for_status()
    with open(HOTPOT_PATH, "wb") as f:
        f.write(r.content)
    print("Download complete.")


class HotpotQADataset(Dataset):
    """
    Wraps HotpotQA multi-hop QA as an ATMAN-compatible dataset.

    Downloads the dataset automatically if not present.
    Converts (question, context, answer) → (input_ids, target_id).
    Elevation and outcome targets are derived heuristically.
    """

    def __init__(self, tokenizer=None, max_len: int = 512, max_examples: int = 5000):
        self.tokenizer = tokenizer
        self.max_len = max_len

        if not HOTPOT_PATH.exists():
            _download_hotpot()

        with open(HOTPOT_PATH) as f:
            all_data = json.load(f)

        # Shuffle for reproducibility, then cap
        random.seed(42)
        random.shuffle(all_data)
        self.data = all_data[:max_examples]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]

        # Build context string (first 3 context docs, 3 sentences each)
        context_parts = []
        for ctx in ex.get("context", [])[:3]:
            if isinstance(ctx, list) and len(ctx) > 1:
                context_parts.extend(str(s) for s in ctx[1][:3])

        text = f"Q: {ex['question']} C: {' '.join(context_parts)}"
        answer = str(ex.get("answer", ""))

        if self.tokenizer is not None:
            input_ids = self.tokenizer.encode(text, add_special_tokens=False)[:self.max_len]
            ans_ids = self.tokenizer.encode(answer, add_special_tokens=False)
            target_id = ans_ids[0] if ans_ids else 0
        else:
            input_ids = _char_encode(text, self.max_len)
            target_id = _char_encode(answer, 1)[0] if answer else 0

        # Heuristic elevation: multi-hop questions benefit from Buddhi
        level = ex.get("level", "easy")
        elevation = 1.0 if level in ("medium", "hard") else 0.0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_id, dtype=torch.long),
            "target_elevation": torch.tensor(elevation, dtype=torch.float32),
            "target_outcome": torch.tensor(0.5, dtype=torch.float32),  # neutral default
        }


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: list[dict]) -> dict:
    """
    Pad input_ids to the longest sequence in the batch.
    Stack scalar targets directly.
    """
    input_ids = [item["input_ids"] for item in batch]
    max_len = max(seq.size(0) for seq in input_ids)

    padded = torch.zeros(len(batch), max_len, dtype=torch.long)
    for i, seq in enumerate(input_ids):
        padded[i, : seq.size(0)] = seq

    out = {
        "input_ids": padded,
        "target_ids": torch.stack([item["target_ids"] for item in batch]),
    }

    # Optional fields: include only if every sample has them
    for key in ("target_elevation", "target_outcome"):
        if all(key in item for item in batch):
            out[key] = torch.stack([item[key] for item in batch])

    return out


# ---------------------------------------------------------------------------
# High-level loader
# ---------------------------------------------------------------------------

def load_datasets(args, tokenizer=None):
    """
    Return (train_loader, val_loader).

    Chooses ATMANDataset if data/training/train.jsonl exists,
    otherwise falls back to HotpotQADataset (downloaded automatically).
    """
    data_dir = Path(getattr(args, "data_dir", "data/training"))
    train_file = data_dir / "train.jsonl"
    max_len = getattr(args, "max_seq_length", 512)
    batch_size = getattr(args, "batch_size", 4)

    if train_file.exists():
        print(f"Loading ATMAN dataset from {train_file}")
        full_dataset = ATMANDataset(str(train_file), tokenizer=tokenizer, max_len=max_len)

        # Use eval.jsonl if present, else 10% split
        eval_file = data_dir / "eval.jsonl"
        if eval_file.exists():
            print(f"Loading eval dataset from {eval_file}")
            train_dataset = full_dataset
            val_dataset = ATMANDataset(str(eval_file), tokenizer=tokenizer, max_len=max_len)
        else:
            val_size = max(1, int(0.1 * len(full_dataset)))
            train_size = len(full_dataset) - val_size
            train_dataset, val_dataset = random_split(
                full_dataset, [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )
    else:
        print("ATMAN data not found — falling back to HotpotQA dataset")
        full_dataset = HotpotQADataset(tokenizer=tokenizer, max_len=max_len)
        val_size = max(1, int(0.1 * len(full_dataset)))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    print(f"Train: {len(train_dataset)} samples ({len(train_loader)} batches)")
    print(f"Val:   {len(val_dataset)} samples ({len(val_loader)} batches)")

    return train_loader, val_loader
