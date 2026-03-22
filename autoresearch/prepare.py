#!/usr/bin/env python3
"""
One-time data preparation for the Samskara-LLM autoresearch loop.

Downloads GSM8K (math reasoning) and ETHICS (moral judgment) datasets,
tokenizes them with tiktoken (cl100k_base — no model download required),
and writes data/meta.pkl so autoresearch/train.py can load data instantly.

Run ONCE before starting the research loop:
    python autoresearch/prepare.py
"""

import math
import pickle
from pathlib import Path

import tiktoken
from datasets import load_dataset

DATA_DIR = Path("data/autoresearch")
META_PATH = DATA_DIR / "meta.pkl"


def download_and_tokenize():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    enc = tiktoken.get_encoding("cl100k_base")
    vocab_size = enc.n_vocab
    print(f"Tokenizer: cl100k_base  vocab_size={vocab_size:,}")

    # ------------------------------------------------------------------
    # GSM8K  (math word problems — trains ElevationRouter)
    # ------------------------------------------------------------------
    print("\nDownloading GSM8K...")
    gsm = load_dataset("openai/gsm8k", "main", trust_remote_code=True)

    gsm_train = []
    for ex in gsm["train"]:
        question = ex["question"].strip()
        answer = ex["answer"].strip()
        text = f"Q: {question}\nA: {answer}"
        ids = enc.encode(text)
        # Heuristic elevation: answers with >3 steps (#### delimiter) → Buddhi
        n_steps = answer.count("\n")
        elevation = 1.0 if n_steps >= 3 else 0.0
        gsm_train.append({"ids": ids, "elevation": elevation, "outcome": 1.0, "source": "gsm8k"})

    gsm_val = []
    for ex in gsm["test"]:
        question = ex["question"].strip()
        answer = ex["answer"].strip()
        text = f"Q: {question}\nA: {answer}"
        ids = enc.encode(text)
        n_steps = answer.count("\n")
        elevation = 1.0 if n_steps >= 3 else 0.0
        gsm_val.append({"ids": ids, "elevation": elevation, "outcome": 1.0, "source": "gsm8k"})

    print(f"  GSM8K train: {len(gsm_train):,}  val: {len(gsm_val):,}")

    # ------------------------------------------------------------------
    # ETHICS — commonsense morality subset (trains Dharma layer)
    # ------------------------------------------------------------------
    print("Downloading ETHICS (commonsense subset)...")
    try:
        ethics = load_dataset("hendrycks/ethics", "commonsense", trust_remote_code=True)

        ethics_train = []
        for ex in ethics["train"]:
            text = f"Scenario: {ex['input'].strip()}"
            ids = enc.encode(text)
            label = float(ex["label"])          # 0=unethical, 1=ethical
            ethics_train.append({"ids": ids, "elevation": label, "outcome": label, "source": "ethics"})

        ethics_val = []
        for ex in ethics["test"]:
            text = f"Scenario: {ex['input'].strip()}"
            ids = enc.encode(text)
            label = float(ex["label"])
            ethics_val.append({"ids": ids, "elevation": label, "outcome": label, "source": "ethics"})

        print(f"  ETHICS train: {len(ethics_train):,}  val: {len(ethics_val):,}")
    except Exception as e:
        print(f"  ETHICS download failed ({e}), skipping.")
        ethics_train, ethics_val = [], []

    # ------------------------------------------------------------------
    # Combine and save
    # ------------------------------------------------------------------
    train_data = gsm_train + ethics_train
    val_data = gsm_val + ethics_val

    # Shuffle train
    import random
    random.seed(42)
    random.shuffle(train_data)

    with open(META_PATH, "wb") as f:
        pickle.dump(
            {
                "vocab_size": vocab_size,
                "encoding": "cl100k_base",
                "train": train_data,
                "val": val_data,
                "train_size": len(train_data),
                "val_size": len(val_data),
            },
            f,
        )

    avg_len = sum(len(d["ids"]) for d in train_data) / max(len(train_data), 1)
    print(f"\nSaved {META_PATH}")
    print(f"  Train: {len(train_data):,} examples  (avg {avg_len:.0f} tokens)")
    print(f"  Val:   {len(val_data):,} examples")
    print(f"\nReady. Run: python autoresearch/train.py")


if __name__ == "__main__":
    download_and_tokenize()
