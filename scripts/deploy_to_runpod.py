#!/usr/bin/env python3
"""
Deploy Samskara-LLM to RunPod GPU cloud.

Two modes:
  --mode train        Full training run (Qwen/Llama base + Samskara layers)
  --mode autoresearch Karpathy-style overnight agent research loop

Usage:
    # Dry-run cost estimate (no API calls)
    python scripts/deploy_to_runpod.py --gpu RTX-4090 --mode autoresearch --hours 8 --dry-run

    # Deploy autoresearch overnight (~$4-6 total)
    RUNPOD_API_KEY=xxx ANTHROPIC_API_KEY=sk-ant-xxx \\
        python scripts/deploy_to_runpod.py --gpu RTX-4090 --mode autoresearch --hours 8

    # Deploy full training (existing behavior)
    RUNPOD_API_KEY=xxx HF_TOKEN=xxx \\
        python scripts/deploy_to_runpod.py --gpu RTX-4090 --mode train --model Qwen/Qwen2.5-1.5B-Instruct
"""

import argparse
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

import requests

# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# GPU types and pricing
GPU_TYPES = {
    "RTX-4090": {"id": "NVIDIA GeForce RTX 4090", "price_per_hr": 0.44, "vram": 24},
    "A40":      {"id": "NVIDIA A40",               "price_per_hr": 0.79, "vram": 48},
    "RTX-A6000":{"id": "NVIDIA RTX A6000",         "price_per_hr": 0.79, "vram": 48},
    "A100-40GB":{"id": "NVIDIA A100-PCIE-40GB",    "price_per_hr": 1.99, "vram": 40},
    "A100":     {"id": "NVIDIA A100 80GB",          "price_per_hr": 2.49, "vram": 80},
}

# Claude API cost estimates (input $/M, output $/M)
CLAUDE_COSTS = {
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "claude-sonnet-4-6":         {"input": 3.00,  "output": 15.00},
}

REPO_URL = "https://github.com/anandin/samskara-llm.git"

# Project root (one level up from scripts/)
_PROJECT_ROOT = Path(__file__).parent.parent


def _bundle_file(rel_path: str) -> str:
    """Read a project file and return its base64-encoded content."""
    return base64.b64encode((_PROJECT_ROOT / rel_path).read_bytes()).decode()


def estimate_costs(gpu_type: str, hours: int, mode: str, agent_model: str) -> dict:
    gpu = GPU_TYPES[gpu_type]
    gpu_cost = gpu["price_per_hr"] * hours

    api_cost = 0.0
    if mode == "autoresearch":
        experiments = (hours * 60) // 5  # one 5-min experiment per slot
        # Each experiment: ~4100 input tokens, ~800 output tokens
        rates = CLAUDE_COSTS.get(agent_model, CLAUDE_COSTS["claude-haiku-4-5-20251001"])
        api_cost = experiments * (4100 * rates["input"] + 800 * rates["output"]) / 1_000_000

    return {
        "gpu_cost": gpu_cost,
        "api_cost": api_cost,
        "total": gpu_cost + api_cost,
        "experiments": (hours * 60) // 5 if mode == "autoresearch" else None,
    }


# Python bootstrap script executed on the pod via dockerArgs.
# Reads source files from env vars (base64), writes them, then runs the loop.
_AUTORESEARCH_BOOTSTRAP = """\
import base64, os, pathlib, subprocess, sys
os.chdir('/workspace')
for path, key in [
    ('autoresearch/prepare.py', 'FILE_PREPARE_PY'),
    ('autoresearch/run.py',     'FILE_RUN_PY'),
    ('autoresearch/train.py',   'FILE_TRAIN_PY'),
    ('autoresearch/program.md', 'FILE_PROGRAM_MD'),
]:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(base64.b64decode(os.environ[key]))
subprocess.run([sys.executable,'-m','pip','install','datasets','tiktoken','anthropic','-q'], check=True)
subprocess.run([sys.executable,'autoresearch/prepare.py'], check=True)
subprocess.run([sys.executable,'autoresearch/run.py',
    '--hours', os.environ.get('RESEARCH_HOURS','8'),
    '--model', os.environ.get('AGENT_MODEL','claude-haiku-4-5-20251001')], check=True)
"""


def create_pod_config(
    gpu_type: str,
    mode: str,
    model_name: str,
    training_config: str,
    hours: int,
    agent_model: str,
    cloud_type: str = "COMMUNITY",
) -> dict:
    gpu = GPU_TYPES[gpu_type]

    env = [
        {"key": "MODEL_NAME",      "value": model_name},
        {"key": "TRAINING_CONFIG", "value": training_config},
        {"key": "HF_TOKEN",        "value": os.getenv("HF_TOKEN", "")},
        {"key": "WANDB_API_KEY",   "value": os.getenv("WANDB_API_KEY", "")},
    ]

    if mode == "autoresearch":
        bootstrap_b64 = base64.b64encode(_AUTORESEARCH_BOOTSTRAP.encode()).decode()
        env += [
            {"key": "ANTHROPIC_API_KEY", "value": os.getenv("ANTHROPIC_API_KEY", "")},
            {"key": "RESEARCH_HOURS",    "value": str(hours)},
            {"key": "AGENT_MODEL",       "value": agent_model},
            # Bootstrap + source files (decoded on the pod, no git clone needed)
            {"key": "B",               "value": bootstrap_b64},
            {"key": "FILE_PREPARE_PY", "value": _bundle_file("autoresearch/prepare.py")},
            {"key": "FILE_RUN_PY",     "value": _bundle_file("autoresearch/run.py")},
            {"key": "FILE_TRAIN_PY",   "value": _bundle_file("autoresearch/train.py")},
            {"key": "FILE_PROGRAM_MD", "value": _bundle_file("autoresearch/program.md")},
        ]
        docker_args = "python3 -c \"import base64,os;exec(base64.b64decode(os.environ['B']).decode())\""
    else:
        docker_args = ""

    return {
        "cloudType":         cloud_type,
        "gpuCount":          1,
        "volumeInGb":        50,
        "containerDiskInGb": 30,
        "minVcpuCount":      4,
        "minMemoryInGb":     16,
        "gpuTypeId":         gpu["id"],
        "name":              f"samskara-{mode}-{gpu_type.lower()}",
        "imageName":         "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
        "ports":             "8888/http,6006/http,22/tcp",
        "volumeMountPath":   "/workspace",
        "env":               env,
        "dockerArgs":        docker_args,
    }


def deploy_to_runpod(config: dict) -> tuple[str, str]:
    if not RUNPOD_API_KEY:
        print("RUNPOD_API_KEY not set")
        print("  Get your key at: https://www.runpod.io/console/settings")
        sys.exit(1)

    query = """
    mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            machine {
                podHostId
            }
        }
    }
    """

    response = requests.post(
        RUNPOD_API_URL,
        json={"query": query, "variables": {"input": config}},
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
        timeout=30,
    )

    if response.status_code != 200:
        print(f"API error: {response.status_code}")
        print(response.text)
        sys.exit(1)

    data = response.json()
    if "errors" in data:
        print(f"GraphQL error: {data['errors']}")
        sys.exit(1)

    pod = data["data"]["podFindAndDeployOnDemand"]
    return pod["id"], pod["machine"]["podHostId"]


def apply_phase_defaults(args):
    """
    --phase shortcuts: pre-set GPU, mode, hours for the three budget phases.

      Phase 1  Autoresearch overnight  RTX-4090   8h   ~$4-6
      Phase 2  Synthetic ATMAN data    (CPU only, no GPU pod needed)
      Phase 3  Full Qwen2.5-7B train   A100-80GB  24h  ~$60

    Any explicit flag overrides the phase default.
    """
    if args.phase is None:
        return

    phase_defaults = {
        1: dict(mode="autoresearch", gpu="RTX-4090",  hours=8,
                model="Qwen/Qwen2.5-1.5B-Instruct"),
        3: dict(mode="train",        gpu="A100",       hours=24,
                model="Qwen/Qwen2.5-7B-Instruct"),
    }

    if args.phase == 2:
        print("Phase 2 — synthetic data generation.")
        print("This only needs the Anthropic API, not a GPU pod.")
        print()
        print("Run locally or on any machine with ANTHROPIC_API_KEY set:")
        print("  python scripts/generate_atman_data.py --n 1000 --source all")
        print()
        print("Cost: ~$2.80 (Haiku, 1000 records)")
        sys.exit(0)

    defaults = phase_defaults.get(args.phase, {})
    for key, val in defaults.items():
        # Only apply if the user didn't set it explicitly
        if getattr(args, key.replace("-", "_")) == parser.get_default(key.replace("-", "_")):
            setattr(args, key.replace("-", "_"), val)

    print(f"Phase {args.phase} defaults applied: "
          f"--mode {args.mode} --gpu {args.gpu} --hours {args.hours}")
    print()


def main():
    global parser
    parser = argparse.ArgumentParser(description="Deploy Samskara-LLM to RunPod")
    parser.add_argument("--phase",  type=int, choices=[1, 2, 3], default=None,
                        help="Budget phase shortcut: 1=autoresearch RTX4090 8h, "
                             "2=data gen (CPU, prints command), 3=full train A100 24h")
    parser.add_argument("--mode",   choices=["train", "autoresearch"], default="autoresearch",
                        help="Deployment mode")
    parser.add_argument("--gpu",    choices=list(GPU_TYPES.keys()), default="RTX-4090",
                        help="GPU type")
    parser.add_argument("--cloud",  choices=["COMMUNITY", "SECURE"], default="COMMUNITY",
                        help="RunPod cloud type (SECURE has better availability, higher cost)")
    parser.add_argument("--hours",  type=int, default=8,
                        help="Run duration in hours")
    parser.add_argument("--model",  default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="Base model for training mode")
    parser.add_argument("--config", default="configs/poc.yaml",
                        help="Training config (train mode only)")
    parser.add_argument("--agent-model", default="claude-haiku-4-5-20251001",
                        choices=list(CLAUDE_COSTS.keys()),
                        help="Claude model for autoresearch agent (autoresearch mode only)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and cost estimate without deploying")
    args = parser.parse_args()

    apply_phase_defaults(args)

    gpu = GPU_TYPES[args.gpu]
    costs = estimate_costs(args.gpu, args.hours, args.mode, args.agent_model)

    print(f"Mode:  {args.mode}")
    print(f"GPU:   {args.gpu}  ({gpu['vram']}GB VRAM)  ${gpu['price_per_hr']:.2f}/hr")
    print(f"Hours: {args.hours}h")
    if args.mode == "autoresearch":
        print(f"Agent: {args.agent_model}")
        print(f"       ~{costs['experiments']} experiments")
    print()
    print(f"  GPU cost:   ${costs['gpu_cost']:.2f}")
    if args.mode == "autoresearch":
        print(f"  API cost:   ${costs['api_cost']:.2f}  (Claude {args.agent_model})")
    print(f"  TOTAL:      ${costs['total']:.2f}")
    print()

    config = create_pod_config(
        args.gpu, args.mode, args.model, args.config, args.hours, args.agent_model,
        cloud_type=args.cloud
    )

    if args.dry_run:
        print("Dry run — pod config:")
        # Truncate large base64 env var values for readability
        config_display = dict(config)
        config_display["env"] = [
            {**e, "value": e["value"][:40] + "…"} if len(e.get("value", "")) > 40 else e
            for e in config_display.get("env", [])
        ]
        print(json.dumps(config_display, indent=2))
        print()
        print("dockerArgs:", config.get("dockerArgs", ""))
        return

    pod_id, host_id = deploy_to_runpod(config)

    print(f"Pod created: {pod_id}")
    print(f"Host:        {host_id}")
    print(f"Monitor:     https://www.runpod.io/console/pods")
    print()
    print(f"Logs:  runpodctl logs {pod_id}")
    if args.mode == "autoresearch":
        print(f"Results will be in autoresearch/run_log.txt on the pod.")
        print(f"SSH in to retrieve: runpodctl exec {pod_id} -- cat autoresearch/run_log.txt")


if __name__ == "__main__":
    main()
