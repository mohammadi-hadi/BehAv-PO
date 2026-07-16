#!/usr/bin/env python3
"""
Phase 2: DPO Preference Fine-Tuning (Mars-PO Component)
========================================================
Fine-tunes each SFT model with DPO preference pairs using OpenAI's API.
Sharpens each agent's cluster-specific decision boundary.

Requires Phase 1 (SFT) to be completed first.

Usage:
    python training/dpo_finetune.py                     # DPO all 3
    python training/dpo_finetune.py conservative        # DPO one
    python training/dpo_finetune.py --status             # check statuses
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MODELS_DIR = "artifacts/model_ids"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_sft_model_ids():
    """Load SFT model IDs from Phase 1."""
    path = os.path.join(MODELS_DIR, "sft_model_ids.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Complete Phase 1 (SFT) first: "
            "python training/sft_finetune.py --status"
        )
    with open(path) as f:
        return json.load(f)


def upload_file(client, filepath, purpose="fine-tune"):
    """Upload a JSONL file to OpenAI."""
    print(f"  Uploading {filepath}...")
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"  File ID: {response.id}")
    return response.id


def create_dpo_job(client, cluster_name, sft_model_id):
    """Create a DPO fine-tuning job for one cluster agent."""
    c_lower = cluster_name.lower()
    train_path = str(config.PREFERENCE_DIR / f"dpo_{c_lower}_train.jsonl")
    val_path = str(config.PREFERENCE_DIR / f"dpo_{c_lower}_val.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found. Run data/prepare_preference_data.py first."
        )

    print(f"\n{'='*50}")
    print(f"DPO Fine-Tuning: {cluster_name} Agent")
    print(f"  Base model: {sft_model_id}")
    print(f"{'='*50}")

    # Upload files
    train_file_id = upload_file(client, train_path)
    val_file_id = upload_file(client, val_path) if os.path.exists(val_path) else None

    # Create DPO fine-tuning job
    params = {
        "training_file": train_file_id,
        "model": sft_model_id,
        "method": {
            "type": "dpo",
            "dpo": {
                "hyperparameters": {
                    "beta": config.DPO_BETA,
                },
            },
        },
        "suffix": f"behav-po-dpo-{c_lower}",
    }
    if val_file_id:
        params["validation_file"] = val_file_id

    try:
        job = client.fine_tuning.jobs.create(**params)
        print(f"  Job ID: {job.id}")
        print(f"  Status: {job.status}")
    except Exception as e:
        if "dpo" in str(e).lower() or "method" in str(e).lower():
            print(f"\n  DPO not available via API. Falling back to SFT-based approach.")
            print(f"  Use training/grpo_rejection_sampling.py as alternative.")
            return None
        raise

    # Save job info
    job_info_path = os.path.join(MODELS_DIR, f"dpo_{c_lower}_job.json")
    with open(job_info_path, "w") as f:
        json.dump({
            "job_id": job.id,
            "cluster": cluster_name,
            "base_model": sft_model_id,
            "train_file": train_file_id,
            "val_file": val_file_id,
            "status": job.status,
        }, f, indent=2)
    print(f"  Job info saved: {job_info_path}")

    return job


def check_status(client):
    """Check the status of all DPO fine-tuning jobs."""
    print(f"\n{'='*50}")
    print("DPO Fine-Tuning Job Status")
    print(f"{'='*50}")

    model_ids = {}
    for cluster in config.CLUSTER_NAMES:
        c_lower = cluster.lower()
        job_info_path = os.path.join(MODELS_DIR, f"dpo_{c_lower}_job.json")
        if not os.path.exists(job_info_path):
            print(f"\n  {cluster}: No job found")
            continue

        with open(job_info_path) as f:
            job_info = json.load(f)

        job = client.fine_tuning.jobs.retrieve(job_info["job_id"])
        print(f"\n  {cluster}:")
        print(f"    Job ID: {job.id}")
        print(f"    Status: {job.status}")

        if job.status == "succeeded":
            print(f"    Model ID: {job.fine_tuned_model}")
            model_ids[cluster] = job.fine_tuned_model
            job_info["status"] = "succeeded"
            job_info["fine_tuned_model"] = job.fine_tuned_model
            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)
        elif job.status == "failed":
            print(f"    Error: {job.error}")

    if len(model_ids) == len(config.CLUSTER_NAMES):
        ids_path = os.path.join(MODELS_DIR, "dpo_model_ids.json")
        with open(ids_path, "w") as f:
            json.dump(model_ids, f, indent=2)
        print(f"\n  All DPO models ready! Saved to {ids_path}")

    return model_ids


def main():
    parser = argparse.ArgumentParser(description="DPO fine-tuning for BehAv-PO agents")
    parser.add_argument("cluster", nargs="?", help="Cluster to fine-tune (or 'all')")
    parser.add_argument("--status", action="store_true", help="Check job statuses")
    parser.add_argument("--wait", action="store_true", help="Wait for jobs to complete")
    args = parser.parse_args()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    if args.status:
        check_status(client)
        return

    sft_model_ids = load_sft_model_ids()

    if args.cluster and args.cluster.lower() != "all":
        cluster = args.cluster.capitalize()
        if cluster not in config.CLUSTER_NAMES:
            print(f"Unknown cluster: {cluster}. Choose from: {config.CLUSTER_NAMES}")
            return
        job = create_dpo_job(client, cluster, sft_model_ids[cluster])
        if args.wait and job:
            time.sleep(60)
            check_status(client)
    else:
        jobs = []
        for cluster in config.CLUSTER_NAMES:
            job = create_dpo_job(client, cluster, sft_model_ids[cluster])
            if job:
                jobs.append(job)
        if jobs:
            print(f"\n{len(jobs)} DPO jobs submitted. Run with --status to check progress.")


if __name__ == "__main__":
    main()
