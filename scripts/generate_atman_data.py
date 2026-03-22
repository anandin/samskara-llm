#!/usr/bin/env python3
"""
Synthetic ATMAN training data generator for Samskara-LLM.

Calls the Claude API to generate richly annotated ATMAN records from
seed queries (drawn from GSM8K, ETHICS, and HotpotQA). Each generated
record contains the full cognitive trace:
  - chitta_seeds         : relevant memory seeds
  - manas_output.signals : FEAR, DESIRE, PATTERN, RISK, OPPORTUNITY, NOISE
  - elevation_triggered  : bool — did the query require Buddhi?
  - buddhi_output        : list of options with dharma scores
  - final_response       : the answer
  - outcome_quality      : [-1.0, 1.0] quality estimate

Output: data/training/train.jsonl  (append mode — safe to re-run)

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python scripts/generate_atman_data.py --n 500
    python scripts/generate_atman_data.py --n 1000 --source gsm8k
    python scripts/generate_atman_data.py --n 200  --source ethics
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import anthropic
from datasets import load_dataset

OUTPUT_FILE = Path("data/training/train.jsonl")

ATMAN_PROMPT = """\
You are a Vedic cognitive architecture annotator. Given a query, produce a JSON
object that represents how the Samskara system (Chitta-Manas-Buddhi-Synthesis)
would process it. Output ONLY valid JSON — no prose, no markdown fences.

Schema:
{{
  "query": "<the original query>",
  "domain": "<one of: math, ethics, factual, strategy, planning>",
  "complexity": "<one of: low, medium, high>",
  "chitta_seeds": ["<3-5 relevant background facts or concepts>"],
  "manas_output": {{
    "signals": {{
      "FEAR":        <float 0-1, threat/risk level>,
      "DESIRE":      <float 0-1, approach/reward signal>,
      "PATTERN":     "<brief pattern recognized>",
      "RISK":        <float 0-1, downside risk>,
      "OPPORTUNITY": <float 0-1, upside potential>,
      "NOISE":       <float 0-1, ambiguity/irrelevance>
    }},
    "fast_answer": "<Manas quick intuitive answer>"
  }},
  "elevation_triggered": <true if query requires careful Buddhi reasoning, else false>,
  "buddhi_output": {{
    "options": [
      {{"text": "<option 1>", "dharma_score": <float 0-1>, "reasoning": "<brief>"}},
      {{"text": "<option 2>", "dharma_score": <float 0-1>, "reasoning": "<brief>"}}
    ],
    "recommendation": "<best option chosen by Buddhi>"
  }},
  "synthesis_triggered": <true if Manas and Buddhi recommendations diverge>,
  "final_response": "<the synthesized final answer>",
  "outcome_quality": <float -1 to 1, estimated quality of the final response>
}}

Query: {query}
"""


def fetch_queries(source: str, n: int) -> list[str]:
    """Fetch n seed queries from the specified dataset."""
    queries = []

    if source in ("gsm8k", "all"):
        print("Loading GSM8K...")
        ds = load_dataset("openai/gsm8k", "main", split="train", trust_remote_code=True)
        for ex in ds:
            queries.append(ex["question"])
            if source == "gsm8k" and len(queries) >= n * 2:
                break

    if source in ("ethics", "all"):
        print("Loading ETHICS...")
        try:
            ds = load_dataset("hendrycks/ethics", "commonsense", split="train", trust_remote_code=True)
            for ex in ds:
                queries.append(ex["input"])
                if source == "ethics" and len(queries) >= n * 2:
                    break
        except Exception as e:
            print(f"ETHICS load failed: {e}")

    if source in ("hotpotqa", "all"):
        print("Loading HotpotQA...")
        try:
            ds = load_dataset("hotpot_qa", "distractor", split="train", trust_remote_code=True)
            for ex in ds:
                queries.append(ex["question"])
                if source == "hotpotqa" and len(queries) >= n * 2:
                    break
        except Exception as e:
            print(f"HotpotQA load failed: {e}")

    random.seed(42)
    random.shuffle(queries)
    return queries[:n]


def generate_record(client: anthropic.Anthropic, query: str, model: str) -> dict | None:
    """Call Claude to generate one ATMAN-annotated record."""
    prompt = ATMAN_PROMPT.format(query=query)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        # Strip markdown fences if present
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        record = json.loads(text)
        record["query"] = query          # ensure original query is preserved
        return record
    except (json.JSONDecodeError, IndexError, anthropic.APIError) as e:
        print(f"  [skip] {type(e).__name__}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",      type=int, default=200,
                        help="Number of records to generate")
    parser.add_argument("--source", default="all",
                        choices=["gsm8k", "ethics", "hotpotqa", "all"],
                        help="Source dataset for seed queries")
    parser.add_argument("--model",  default="claude-haiku-4-5-20251001",
                        help="Anthropic model (Haiku is fast and cheap for data gen)")
    parser.add_argument("--sleep",  type=float, default=0.5,
                        help="Seconds to sleep between API calls (rate limiting)")
    args = parser.parse_args()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    client = anthropic.Anthropic()    # reads ANTHROPIC_API_KEY from env

    print(f"Fetching {args.n} seed queries from: {args.source}")
    queries = fetch_queries(args.source, args.n)
    print(f"Got {len(queries)} queries. Generating ATMAN annotations...\n")

    generated = 0
    skipped   = 0

    with open(OUTPUT_FILE, "a") as out_f:
        for i, query in enumerate(queries):
            record = generate_record(client, query, args.model)
            if record is not None:
                out_f.write(json.dumps(record) + "\n")
                out_f.flush()
                generated += 1
                if generated % 10 == 0:
                    print(f"  [{generated}/{args.n}] generated  ({skipped} skipped)")
            else:
                skipped += 1

            if generated >= args.n:
                break

            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"\nDone. Generated {generated} records ({skipped} skipped).")
    print(f"Output: {OUTPUT_FILE}")
    print(f"\nTo train with this data:")
    print(f"  python training/train.py --config configs/poc.yaml --data-dir data/training/")


if __name__ == "__main__":
    main()
