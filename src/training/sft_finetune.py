#!/usr/bin/env python3
"""
Phase 1: SFT Fine-Tuning
=========================
Fine-tunes 3 GPT-4o-mini models (one per cluster) using OpenAI's API.
Each model learns its cluster's annotation behavior from the SFT dataset.

Usage:
    python training/sft_finetune.py                    # fine-tune all 3
    python training/sft_finetune.py conservative       # fine-tune one
    python training/sft_finetune.py --status            # check job statuses
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


def upload_file(client, filepath, purpose="fine-tune"):
    """Upload a JSONL file to OpenAI."""
    print(f"  Uploading {filepath}...")
    with open(filepath, "rb") as f:
        response = client.files.create(file=f, purpose=purpose)
    print(f"  File ID: {response.id}")
    return response.id


def create_sft_job(client, cluster_name):
    """Create an SFT fine-tuning job for one cluster agent."""
    c_lower = cluster_name.lower()
    train_path = str(config.PREFERENCE_DIR / f"sft_{c_lower}_train.jsonl")
    val_path = str(config.PREFERENCE_DIR / f"sft_{c_lower}_val.jsonl")

    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"{train_path} not found. Run data/prepare_preference_data.py first."
        )

    print(f"\n{'='*50}")
    print(f"SFT Fine-Tuning: {cluster_name} Agent")
    print(f"{'='*50}")

    # Upload files
    train_file_id = upload_file(client, train_path)
    val_file_id = upload_file(client, val_path) if os.path.exists(val_path) else None

    # Create fine-tuning job
    params = {
        "training_file": train_file_id,
        "model": config.BASE_MODEL,
        "hyperparameters": {
            "n_epochs": config.SFT_EPOCHS,
        },
        "suffix": f"behav-po-sft-{c_lower}",
    }
    if val_file_id:
        params["validation_file"] = val_file_id

    job = client.fine_tuning.jobs.create(**params)
    print(f"  Job ID: {job.id}")
    print(f"  Status: {job.status}")

    # Save job info
    job_info_path = os.path.join(MODELS_DIR, f"sft_{c_lower}_job.json")
    with open(job_info_path, "w") as f:
        json.dump({
            "job_id": job.id,
            "cluster": cluster_name,
            "model": config.BASE_MODEL,
            "train_file": train_file_id,
            "val_file": val_file_id,
            "status": job.status,
        }, f, indent=2)
    print(f"  Job info saved: {job_info_path}")

    return job


def check_status(client):
    """Check the status of all SFT fine-tuning jobs."""
    print(f"\n{'='*50}")
    print("SFT Fine-Tuning Job Status")
    print(f"{'='*50}")

    model_ids = {}
    for cluster in config.CLUSTER_NAMES:
        c_lower = cluster.lower()
        job_info_path = os.path.join(MODELS_DIR, f"sft_{c_lower}_job.json")
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
            # Update job info
            job_info["status"] = "succeeded"
            job_info["fine_tuned_model"] = job.fine_tuned_model
            with open(job_info_path, "w") as f:
                json.dump(job_info, f, indent=2)
        elif job.status == "failed":
            print(f"    Error: {job.error}")

    # Save model IDs if all done
    if len(model_ids) == len(config.CLUSTER_NAMES):
        ids_path = os.path.join(MODELS_DIR, "sft_model_ids.json")
        with open(ids_path, "w") as f:
            json.dump(model_ids, f, indent=2)
        print(f"\n  All models ready! Saved to {ids_path}")

    return model_ids


def wait_for_jobs(client, jobs, poll_interval=60):
    """Poll until all jobs complete."""
    print(f"\nWaiting for {len(jobs)} jobs to complete...")
    pending = {j.id: j for j in jobs}

    while pending:
        time.sleep(poll_interval)
        for job_id in list(pending.keys()):
            job = client.fine_tuning.jobs.retrieve(job_id)
            if job.status in ("succeeded", "failed", "cancelled"):
                print(f"  {job_id}: {job.status}")
                if job.status == "succeeded":
                    print(f"    Model: {job.fine_tuned_model}")
                del pending[job_id]
            else:
                print(f"  {job_id}: {job.status}...")

    check_status(client)


def main():
    parser = argparse.ArgumentParser(description="SFT fine-tuning for BehAv-PO agents")
    parser.add_argument("cluster", nargs="?", help="Cluster to fine-tune (or 'all')")
    parser.add_argument("--status", action="store_true", help="Check job statuses")
    parser.add_argument("--wait", action="store_true", help="Wait for jobs to complete")
    args = parser.parse_args()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    if args.status:
        check_status(client)
        return

    if args.cluster and args.cluster.lower() != "all":
        # Fine-tune one cluster
        cluster = args.cluster.capitalize()
        if cluster not in config.CLUSTER_NAMES:
            print(f"Unknown cluster: {cluster}. Choose from: {config.CLUSTER_NAMES}")
            return
        job = create_sft_job(client, cluster)
        if args.wait:
            wait_for_jobs(client, [job])
    else:
        # Fine-tune all 3
        jobs = []
        for cluster in config.CLUSTER_NAMES:
            job = create_sft_job(client, cluster)
            jobs.append(job)
        if args.wait:
            wait_for_jobs(client, jobs)
        else:
            print(f"\n{len(jobs)} jobs submitted. Run with --status to check progress.")


if __name__ == "__main__":
    main()
