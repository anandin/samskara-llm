#!/usr/bin/env python3
"""
Deploy Samskara-LLM training to Lambda Labs GPU cloud.

Usage:
    python scripts/deploy_to_lambda.py --gpu A10 --model llama-3.1-8b
"""

import argparse
import json
import os
import sys

# Lambda Labs API configuration
LAMBDA_API_KEY = os.getenv("LAMBDA_API_KEY", "")
LAMBDA_API_URL = "https://cloud.lambdalabs.com/api/v1"

# GPU types
GPU_TYPES = {
    "A100": {"name": "gpu_1x_a100_sxm4", "price_per_hr": 1.99, "vram": 80},
    "A10": {"name": "gpu_1x_a10", "price_per_hr": 0.60, "vram": 24},
    "RTX-A6000": {"name": "gpu_1x_rtx6000", "price_per_hr": 0.50, "vram": 48},
    "H100": {"name": "gpu_1x_h100_pcie", "price_per_hr": 2.49, "vram": 80},
}


def create_instance_config(
    gpu_type: str,
    model_name: str,
    training_config: str,
) -> dict:
    """Create Lambda Labs instance configuration."""
    
    gpu = GPU_TYPES.get(gpu_type, GPU_TYPES["A100"])
    
    config = {
        "instance_type": gpu["name"],
        "ssh_key_names": ["samskara-key"],  # Add your SSH key name
        "file_system_names": [],
        "quantity": 1,
        "name": f"samskara-llm-{model_name.split('/')[-1].lower()}",
    }
    
    return config


def deploy_to_lambda(config: dict) -> str:
    """Deploy instance to Lambda Labs."""
    
    if not LAMBDA_API_KEY:
        print("❌ LAMBDA_API_KEY not set")
        print("   Get your API key from: https://cloud.lambdalabs.com/api-keys")
        sys.exit(1)
    
    import requests
    
    # Launch instance
    response = requests.post(
        f"{LAMBDA_API_URL}/instance-operations/launch",
        json=config,
        headers={"Authorization": f"Bearer {LAMBDA_API_KEY}"},
    )
    
    if response.status_code != 200:
        print(f"❌ API error: {response.status_code}")
        print(response.text)
        sys.exit(1)
    
    data = response.json()
    
    if "error" in data:
        print(f"❌ API error: {data['error']}")
        sys.exit(1)
    
    instance_ids = data.get("instance_ids", [])
    return instance_ids[0] if instance_ids else None


def main():
    parser = argparse.ArgumentParser(description="Deploy Samskara-LLM training to Lambda Labs")
    parser.add_argument("--gpu", choices=list(GPU_TYPES.keys()), default="A10",
                       help="GPU type")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct",
                       help="Base model to train")
    parser.add_argument("--config", default="configs/poc.yaml",
                       help="Training config file")
    parser.add_argument("--dry-run", action="store_true",
                       help="Print config without deploying")
    
    args = parser.parse_args()
    
    print(f"🚀 Deploying Samskara-LLM training to Lambda Labs")
    print(f"   GPU: {args.gpu}")
    print(f"   Model: {args.model}")
    print(f"   Config: {args.config}")
    print()
    
    config = create_instance_config(args.gpu, args.model, args.config)
    
    if args.dry_run:
        print("📋 Configuration (dry run):")
        print(json.dumps(config, indent=2))
        return
    
    try:
        instance_id = deploy_to_lambda(config)
        
        print(f"✅ Instance created successfully!")
        print(f"   Instance ID: {instance_id}")
        print()
        print(f"🔗 Monitor at: https://cloud.lambdalabs.com/instances")
        print()
        print(f"⏱️  Current pricing: ${GPU_TYPES[args.gpu]['price_per_hr']}/hr")
        
    except Exception as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
