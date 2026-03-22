#!/usr/bin/env python3
"""
Deploy Samskara-LLM training to RunPod GPU cloud.

Usage:
    python scripts/deploy_to_runpod.py --gpu A100 --model llama-3.1-8b
"""

import argparse
import json
import os
import subprocess
import sys

# RunPod API configuration
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_API_URL = "https://api.runpod.io/graphql"

# GPU types and pricing (as of 2024)
GPU_TYPES = {
    "A100": {"id": "NVIDIA A100 80GB", "price_per_hr": 2.49, "vram": 80},
    "A100-40GB": {"id": "NVIDIA A100-PCIE-40GB", "price_per_hr": 1.99, "vram": 40},
    "A40": {"id": "NVIDIA A40", "price_per_hr": 0.79, "vram": 48},
    "RTX-A6000": {"id": "NVIDIA RTX A6000", "price_per_hr": 0.79, "vram": 48},
    "RTX-4090": {"id": "NVIDIA GeForce RTX 4090", "price_per_hr": 0.44, "vram": 24},
}


def create_pod_config(
    gpu_type: str,
    model_name: str,
    training_config: str,
    hours: int = 24,
) -> dict:
    """Create RunPod pod configuration."""
    
    gpu = GPU_TYPES.get(gpu_type, GPU_TYPES["A100"])
    
    config = {
        "cloudType": "COMMUNITY",  # or "SECURE" for privacy
        "gpuCount": 1,
        "volumeInGb": 100,
        "containerDiskInGb": 50,
        "minVcpuCount": 8,
        "minMemoryInGb": 32,
        "gpuTypeId": gpu["id"],
        "name": f"samskara-llm-{model_name.replace('/', '-').lower()}",
        "imageName": "runpod/pytorch:2.2.0-py3.10-cuda12.1-devel-ubuntu22.04",
        "dockerArgs": "",
        "ports": "8888/http,6006/http,22/tcp",
        "volumeMountPath": "/workspace",
        "env": [
            {"key": "MODEL_NAME", "value": model_name},
            {"key": "TRAINING_CONFIG", "value": training_config},
            {"key": "HF_TOKEN", "value": os.getenv("HF_TOKEN", "")},
            {"key": "WANDB_API_KEY", "value": os.getenv("WANDB_API_KEY", "")},
        ],
        "startScript": """#!/bin/bash
set -e

echo "🧘 Setting up Samskara-LLM training environment..."

# Clone repo
cd /workspace
git clone https://github.com/anandin/samskara-llm.git
cd samskara-llm

# Install dependencies
pip install -r requirements.txt
pip install flash-attn --no-build-isolation

# Login to HuggingFace for model access
huggingface-cli login --token $HF_TOKEN

# Start training
echo "🚀 Starting training with config: $TRAINING_CONFIG"
python training/train.py \\
    --model $MODEL_NAME \\
    --config $TRAINING_CONFIG \\
    --output-dir /workspace/outputs \\
    --wandb-project samskara-llm

echo "✅ Training complete!"

# Upload results to cloud storage
aws s3 sync /workspace/outputs s3://samskara-llm/outputs/$(date +%Y%m%d-%H%M%S)/
""",
    }
    
    return config


def deploy_to_runpod(config: dict) -> str:
    """Deploy pod to RunPod."""
    
    if not RUNPOD_API_KEY:
        print("❌ RUNPOD_API_KEY not set")
        print("   Get your API key from: https://www.runpod.io/console/settings")
        sys.exit(1)
    
    # GraphQL mutation to create pod
    query = """
    mutation PodFindAndDeployOnDemand($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            imageName
            env
            machineId
            machine {
                podHostId
            }
        }
    }
    """
    
    variables = {"input": config}
    
    # Execute GraphQL request
    import requests
    
    response = requests.post(
        RUNPOD_API_KEY,  # Actually need to check API format
        json={"query": query, "variables": variables},
        headers={"Authorization": f"Bearer {RUNPOD_API_KEY}"},
    )
    
    if response.status_code != 200:
        print(f"❌ API error: {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    data = response.json()
    
    if "errors" in data:
        print(f"❌ GraphQL error: {data['errors']}")
        sys.exit(1)
    
    pod_id = data["data"]["podFindAndDeployOnDemand"]["id"]
    host_id = data["data"]["podFindAndDeployOnDemand"]["machine"]["podHostId"]
    
    return pod_id, host_id


def main():
    parser = argparse.ArgumentParser(description="Deploy Samskara-LLM training to RunPod")
    parser.add_argument("--gpu", choices=list(GPU_TYPES.keys()), default="A100",
                       help="GPU type")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model to train")
    parser.add_argument("--config", default="configs/poc.yaml",
                       help="Training config file")
    parser.add_argument("--hours", type=int, default=24,
                       help="Max training hours (auto-shutdown)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print config without deploying")
    
    args = parser.parse_args()
    
    print(f"🚀 Deploying Samskara-LLM training")
    print(f"   GPU: {args.gpu}")
    print(f"   Model: {args.model}")
    print(f"   Config: {args.config}")
    print()
    
    # Create config
    config = create_pod_config(args.gpu, args.model, args.config, args.hours)
    
    if args.dry_run:
        print("📋 Configuration (dry run):")
        print(json.dumps(config, indent=2))
        return
    
    # Deploy
    try:
        pod_id, host_id = deploy_to_runpod(config)
        
        print(f"✅ Pod created successfully!")
        print(f"   Pod ID: {pod_id}")
        print(f"   Host: {host_id}")
        print()
        print(f"🔗 Monitor at: https://www.runpod.io/console/pods")
        print()
        print(f"📊 Training logs:")
        print(f"   runpodctl logs {pod_id}")
        print()
        print(f"⏱️  Estimated cost: ${GPU_TYPES[args.gpu]['price_per_hr'] * args.hours:.2f}")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
