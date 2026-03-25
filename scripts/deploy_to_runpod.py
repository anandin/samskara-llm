#!/usr/bin/env python3
"""
Deploy SamskaraLLM jobs to RunPod with proper lifecycle management.

Fixes the auto-restart bug: pods now commit results and stop themselves
after job completion instead of restarting indefinitely.

Usage:
    # Phase 2: data generation on CPU pod
    python scripts/deploy_to_runpod.py \\
        --job "python scripts/generate_atman_data.py --count 1000" \\
        --gpu none --name atman-datagen

    # Phase 3: full model training on A100
    python scripts/deploy_to_runpod.py \\
        --job "python train.py" \\
        --gpu A100_80GB --name samskara-train

    # Preview startup script without launching
    python scripts/deploy_to_runpod.py --dry-run \\
        --job "python scripts/generate_atman_data.py"

Env vars:
    RUNPOD_API_KEY   — required (unless --dry-run)
    GITHUB_TOKEN     — optional, for private repo cloning
"""

import argparse
import json
import os
import sys
import textwrap

REPO_URL = "https://github.com/anandin/samskara-llm.git"
DEFAULT_BRANCH = "main"

GPU_CONFIGS = {
    "none": {
        "gpu_type_id": None,
        "cloud_type": "COMMUNITY",
        "gpu_count": 0,
        "description": "CPU-only pod (for data generation, ~$0.10/hr)",
        "template_id": "runpod-torch-2",
    },
    "A100_80GB": {
        "gpu_type_id": "NVIDIA A100 80GB PCIe",
        "cloud_type": "SECURE",
        "gpu_count": 1,
        "description": "A100 80GB (for full model training, ~$1.50/hr)",
        "template_id": "runpod-torch-2",
    },
    "A100_40GB": {
        "gpu_type_id": "NVIDIA A100-SXM4-40GB",
        "cloud_type": "COMMUNITY",
        "gpu_count": 1,
        "description": "A100 40GB (budget training, ~$1.00/hr)",
        "template_id": "runpod-torch-2",
    },
    "RTX_4090": {
        "gpu_type_id": "NVIDIA GeForce RTX 4090",
        "cloud_type": "COMMUNITY",
        "gpu_count": 1,
        "description": "RTX 4090 (smaller jobs, ~$0.40/hr)",
        "template_id": "runpod-torch-2",
    },
}


def generate_startup_script(job_command, branch, repo_url, env_vars=None):
    """Generate the bash startup script that runs inside the pod.

    Key fix: after the job command exits, the script commits any outputs
    and stops the pod — preventing the auto-restart loop.
    """
    env_exports = ""
    if env_vars:
        for k, v in env_vars.items():
            env_exports += f'export {k}="{v}"\n'

    script = textwrap.dedent(f"""\
        #!/bin/bash
        set -euo pipefail

        echo "========================================="
        echo "SamskaraLLM RunPod Job Starting"
        echo "Branch: {branch}"
        echo "Job:    {job_command}"
        echo "========================================="

        # --- Setup ---
        cd /workspace

        # Clone repo
        if [ -d "samskara-llm" ]; then
            cd samskara-llm
            git fetch origin {branch}
            git checkout {branch}
            git pull origin {branch}
        else
            git clone -b {branch} {repo_url} samskara-llm
            cd samskara-llm
        fi

        # Install dependencies
        pip install -q -r requirements.txt 2>/dev/null || true

        # Set env vars
        {env_exports}

        # --- SIGTERM handler for graceful shutdown ---
        cleanup() {{
            echo ""
            echo "========================================="
            echo "SIGTERM received — saving progress..."
            echo "========================================="
            cd /workspace/samskara-llm
            git add -A
            git diff --cached --quiet || git commit -m "auto: interrupted job - saving progress"
            git push origin {branch} || echo "WARNING: git push failed"
            echo "Progress saved. Exiting."
            exit 0
        }}
        trap cleanup SIGTERM SIGINT

        # --- Run the job ---
        echo ""
        echo "========================================="
        echo "Running: {job_command}"
        echo "========================================="

        {job_command}
        JOB_EXIT_CODE=$?

        echo ""
        echo "========================================="
        echo "Job finished with exit code: $JOB_EXIT_CODE"
        echo "========================================="

        # --- AUTO-STOP FIX ---
        # This is the critical fix for the auto-restart bug.
        # After the job completes, commit results and stop the pod.
        # Without this, RunPod restarts the pod and re-runs the job.

        cd /workspace/samskara-llm
        git add -A
        git diff --cached --quiet || git commit -m "auto: job complete (exit=$JOB_EXIT_CODE)"
        git push origin {branch} || echo "WARNING: git push failed, results may be lost"

        echo "Results committed and pushed to {branch}"

        # Stop the pod to prevent auto-restart and idle billing
        if [ -n "${{RUNPOD_POD_ID:-}}" ]; then
            echo "Stopping pod $RUNPOD_POD_ID..."
            runpodctl stop pod $RUNPOD_POD_ID
        else
            echo "WARNING: RUNPOD_POD_ID not set, cannot auto-stop pod."
            echo "Pod will remain running. Stop manually to avoid billing."
        fi
    """)
    return script


def create_pod(api_key, name, gpu_config, startup_script, disk_size_gb=20):
    """Create a RunPod pod via the API."""
    try:
        import runpod
    except ImportError:
        print("ERROR: runpod package not installed. Run: pip install runpod")
        sys.exit(1)

    runpod.api_key = api_key

    config = GPU_CONFIGS[gpu_config]

    pod_config = {
        "name": name,
        "image_name": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
        "disk_size_in_gb": disk_size_gb,
        "volume_in_gb": 0,
        "docker_args": f'bash -c \'{startup_script}\'',
    }

    if config["gpu_type_id"]:
        pod_config["gpu_type_id"] = config["gpu_type_id"]
        pod_config["gpu_count"] = config["gpu_count"]
        pod_config["cloud_type"] = config["cloud_type"]
    else:
        # CPU-only: use the smallest available GPU but don't actually need it
        # RunPod requires at least one GPU — use cheapest option
        pod_config["gpu_type_id"] = "NVIDIA GeForce RTX 3070"
        pod_config["gpu_count"] = 1
        pod_config["cloud_type"] = "COMMUNITY"

    print(f"Creating pod '{name}' with {gpu_config}...")
    pod = runpod.create_pod(**pod_config)
    return pod


def main():
    parser = argparse.ArgumentParser(
        description="Deploy SamskaraLLM jobs to RunPod",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--job", type=str, required=True,
                        help="Command to run inside the pod (e.g., 'python train.py')")
    parser.add_argument("--gpu", type=str, default="A100_80GB",
                        choices=list(GPU_CONFIGS.keys()),
                        help="GPU type (default: A100_80GB)")
    parser.add_argument("--name", type=str, default="samskara-job",
                        help="Pod name (default: samskara-job)")
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH,
                        help=f"Git branch to use (default: {DEFAULT_BRANCH})")
    parser.add_argument("--disk-size", type=int, default=20,
                        help="Disk size in GB (default: 20)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print startup script without creating pod")
    parser.add_argument("--env", nargs="*", default=[],
                        help="Environment variables as KEY=VALUE pairs")
    args = parser.parse_args()

    # Parse env vars
    env_vars = {}
    for ev in args.env:
        if "=" not in ev:
            print(f"ERROR: Invalid env var format '{ev}'. Use KEY=VALUE.")
            sys.exit(1)
        key, value = ev.split("=", 1)
        env_vars[key] = value

    # Generate startup script
    startup_script = generate_startup_script(
        job_command=args.job,
        branch=args.branch,
        repo_url=REPO_URL,
        env_vars=env_vars,
    )

    if args.dry_run:
        print("DRY RUN — Startup script that would run inside the pod:\n")
        print("=" * 60)
        print(startup_script)
        print("=" * 60)
        print(f"\nGPU config: {args.gpu} — {GPU_CONFIGS[args.gpu]['description']}")
        print(f"Pod name:   {args.name}")
        print(f"Branch:     {args.branch}")
        print(f"Disk:       {args.disk_size} GB")
        if env_vars:
            print(f"Env vars:   {list(env_vars.keys())}")
        print("\nKey auto-stop features:")
        print("  - SIGTERM handler saves progress on interruption")
        print("  - Job results committed and pushed after completion")
        print("  - Pod auto-stops via runpodctl after job exits")
        print("  - No auto-restart loop")
        return

    # Check API key
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: RUNPOD_API_KEY environment variable not set.")
        sys.exit(1)

    # Confirm
    config = GPU_CONFIGS[args.gpu]
    print(f"SamskaraLLM RunPod Deployer")
    print(f"  Job:    {args.job}")
    print(f"  GPU:    {args.gpu} — {config['description']}")
    print(f"  Name:   {args.name}")
    print(f"  Branch: {args.branch}")
    print(f"  Disk:   {args.disk_size} GB")
    print()

    response = input("Create pod? [y/N] ")
    if response.lower() not in ("y", "yes"):
        print("Aborted.")
        return

    pod = create_pod(api_key, args.name, args.gpu, startup_script, args.disk_size)
    print(f"\nPod created successfully!")
    print(f"  Pod ID:  {pod.get('id', 'unknown')}")
    print(f"  Status:  {pod.get('desiredStatus', 'unknown')}")
    print(f"\nMonitor at: https://www.runpod.io/console/pods")
    print(f"Results will be auto-pushed to branch '{args.branch}' when the job completes.")


if __name__ == "__main__":
    main()
