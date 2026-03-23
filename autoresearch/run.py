#!/usr/bin/env python3
"""
Samskara-LLM autoresearch agent loop.

Runs the autoresearch cycle autonomously:
  1. Read program.md + train.py
  2. Ask an LLM to propose a modification
  3. Apply the diff to train.py
  4. Run train.py, capture val_bpb
  5. Keep the change if val_bpb improves; revert otherwise
  6. Update program.md with new best score and agent hypothesis
  7. Repeat until --hours elapsed

Usage:
    python autoresearch/run.py                  # 8-hour run (default)
    python autoresearch/run.py --hours 0.1      # ~6-min test (1 experiment)
    python autoresearch/run.py --hours 8 --model claude-sonnet-4-6

Requirements:
    ANTHROPIC_API_KEY environment variable must be set.
    Run prepare.py first if you haven't already.
"""

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import anthropic

TRAIN_PY    = Path("autoresearch/train.py")
PROGRAM_MD  = Path("autoresearch/program.md")
LOG_FILE    = Path("autoresearch/run_log.txt")

SYSTEM_PROMPT = """\
You are an expert ML researcher optimizing the Samskara cognitive architecture.
You will be shown the current train.py and program.md. Your job is to propose
ONE targeted modification to train.py that you believe will reduce val_bpb.

Rules:
- Output ONLY a unified diff (--- a/train.py / +++ b/train.py format)
- Keep changes minimal and focused — one hypothesis per experiment
- Never modify the data loading block or the final val_bpb print line
- n_heads must divide d_model evenly
- After the diff, add one line: HYPOTHESIS: <one sentence explaining your reasoning>
"""


def parse_val_bpb(output: str) -> float | None:
    """Extract val_bpb from train.py stdout."""
    for line in output.splitlines():
        m = re.search(r"val_bpb:\s*([0-9]+\.[0-9]+)", line)
        if m:
            return float(m.group(1))
    return None


def run_experiment() -> tuple[float | None, str]:
    """Run train.py and return (val_bpb, stdout)."""
    result = subprocess.run(
        [sys.executable, str(TRAIN_PY)],
        capture_output=True,
        text=True,
        timeout=360,  # 6-minute hard kill
    )
    output = result.stdout + result.stderr
    return parse_val_bpb(output), output


def apply_diff(diff_text: str) -> bool:
    """Apply a unified diff to train.py. Returns True on success."""
    diff_path = Path("autoresearch/_patch.diff")
    diff_path.write_text(diff_text)
    result = subprocess.run(
        ["patch", "--forward", "--reject-file=-", "-p1", str(TRAIN_PY)],
        input=diff_text,
        capture_output=True,
        text=True,
    )
    diff_path.unlink(missing_ok=True)
    return result.returncode == 0


def revert_train_py(backup: str):
    TRAIN_PY.write_text(backup)


def update_program_md(best_bpb: float, hypothesis: str, experiment: int):
    content = PROGRAM_MD.read_text()

    # Update current best line
    content = re.sub(
        r"val_bpb: .*",
        f"val_bpb: {best_bpb:.4f}",
        content,
        count=1,
    )
    # Update hypothesis line
    content = re.sub(
        r"\[Agent fills this in before each run\]",
        hypothesis,
        content,
    )
    # Append to research log table
    log_line = f"| exp-{experiment:03d} | {hypothesis[:60]} | {best_bpb:.4f} | |"
    content = content.replace(
        "| baseline  | default hyperparams | — | first run |",
        f"| baseline  | default hyperparams | — | first run |\n{log_line}",
    )
    PROGRAM_MD.write_text(content)


def sync_to_hub(hf_repo: str, experiment: int, best_bpb: float):
    """Upload train.py, run_log.txt, program.md to HuggingFace Hub. Never raises."""
    if not hf_repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=os.environ.get("HF_TOKEN", ""))
        api.create_repo(repo_id=hf_repo, repo_type="dataset", exist_ok=True, private=True)
        commit_msg = f"Exp {experiment:03d}: val_bpb={best_bpb:.4f}"
        for path in [TRAIN_PY, LOG_FILE, PROGRAM_MD]:
            if path.exists():
                api.upload_file(
                    path_or_fileobj=str(path),
                    path_in_repo=str(path),
                    repo_id=hf_repo,
                    repo_type="dataset",
                    commit_message=commit_msg,
                )
        print(f"  [sync] → HF:{hf_repo}  ({commit_msg})")
    except Exception as e:
        print(f"  [sync] WARNING: upload failed — {e}")


def ask_agent(client: anthropic.Anthropic, model: str, train_src: str, program_src: str) -> str:
    """Ask the LLM for a proposed diff."""
    user_msg = f"""## program.md
{program_src}

## current train.py
```python
{train_src}
```

Propose ONE modification to reduce val_bpb. Output a unified diff then HYPOTHESIS: ..."""

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    return message.content[0].text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours",      type=float, default=8.0,    help="How long to run (hours)")
    parser.add_argument("--model",      default="claude-sonnet-4-6", help="Anthropic model ID")
    parser.add_argument("--hf-repo",    default="",  help="HuggingFace repo for periodic sync (empty=disabled)")
    parser.add_argument("--sync-every", type=int, default=10,   help="Sync to HF every N experiments")
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    deadline = time.time() + args.hours * 3600
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Establish baseline
    print("=== Running baseline experiment ===")
    baseline_bpb, baseline_out = run_experiment()
    if baseline_bpb is None:
        print("ERROR: baseline run did not produce val_bpb. Check train.py.")
        print(baseline_out)
        sys.exit(1)

    best_bpb   = baseline_bpb
    experiment = 0
    print(f"Baseline val_bpb: {best_bpb:.4f}")

    with open(LOG_FILE, "w") as log:
        log.write(f"Baseline val_bpb: {best_bpb:.4f}\n\n")

    while time.time() < deadline:
        experiment += 1
        print(f"\n=== Experiment {experiment} (best so far: {best_bpb:.4f}) ===")

        train_src   = TRAIN_PY.read_text()
        program_src = PROGRAM_MD.read_text()
        backup      = train_src

        # Ask agent for a proposed change
        try:
            agent_response = ask_agent(client, args.model, train_src, program_src)
        except Exception as e:
            print(f"  Agent API error: {e} — skipping experiment")
            time.sleep(10)
            continue

        # Extract diff and hypothesis
        diff_match = re.search(r"(---.*?(?=HYPOTHESIS:|$))", agent_response, re.DOTALL)
        hyp_match  = re.search(r"HYPOTHESIS:\s*(.+)", agent_response)
        hypothesis = hyp_match.group(1).strip() if hyp_match else "no hypothesis"

        if not diff_match:
            print("  Agent did not produce a diff — skipping")
            continue

        diff_text = diff_match.group(1).strip()
        print(f"  Hypothesis: {hypothesis}")

        # Apply diff
        if not apply_diff(diff_text):
            print("  Patch failed to apply — reverting")
            revert_train_py(backup)
            continue

        # Run experiment
        try:
            val_bpb, output = run_experiment()
        except subprocess.TimeoutExpired:
            print("  Experiment timed out — reverting")
            revert_train_py(backup)
            continue

        if val_bpb is None:
            print("  No val_bpb in output — reverting")
            print(output[-500:])
            revert_train_py(backup)
            continue

        improved = val_bpb < best_bpb
        print(f"  val_bpb: {val_bpb:.4f}  {'✓ IMPROVED' if improved else '✗ no improvement'}")

        with open(LOG_FILE, "a") as log:
            log.write(f"Exp {experiment:03d}: val_bpb={val_bpb:.4f}  improved={improved}\n")
            log.write(f"  hypothesis: {hypothesis}\n\n")

        if improved:
            best_bpb = val_bpb
            update_program_md(best_bpb, hypothesis, experiment)
            print(f"  New best: {best_bpb:.4f} — keeping change")
            sync_to_hub(args.hf_repo, experiment, best_bpb)
        else:
            revert_train_py(backup)
            print("  Reverting to previous train.py")

        if experiment % args.sync_every == 0:
            sync_to_hub(args.hf_repo, experiment, best_bpb)

    print(f"\n=== Done. Best val_bpb: {best_bpb:.4f} after {experiment} experiments ===")
    print(f"Log saved to {LOG_FILE}")
    sync_to_hub(args.hf_repo, experiment, best_bpb)


if __name__ == "__main__":
    main()
