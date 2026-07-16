#!/usr/bin/env python3
"""
Phase 3 + Phase 4 Orchestrator
===============================
Waits for OpenAI queue to free up, submits Phase 3 retrain jobs in batches,
monitors all jobs, evaluates when complete.

Runs in background, no manual intervention needed.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_client():
    from openai import OpenAI
    return OpenAI()


def count_active_jobs(client, model_prefix="gpt-4.1-mini"):
    """Count active (non-terminal) fine-tune jobs on a model."""
    try:
        jobs = client.fine_tuning.jobs.list(limit=30)
        return sum(
            1 for j in jobs.data
            if j.status in ["queued", "validating_files", "running"]
            and model_prefix in (j.model or "")
        )
    except Exception as e:
        print(f"[queue check] Retry: {str(e)[:60]}")
        return 99  # be conservative


def wait_for_slots(client, needed, model_prefix="gpt-4.1-mini"):
    while True:
        active = count_active_jobs(client, model_prefix)
        slots_free = 6 - active
        print(f"[queue] {active}/6 used, {slots_free} free")
        if slots_free >= needed:
            return
        time.sleep(120)


def wait_for_phase3_sampling():
    """Wait for raw_samples.json to exist."""
    while not os.path.exists("data/preference/phase3_raw_samples.json"):
        print("[phase3] Waiting for sampling to complete...")
        time.sleep(60)
    print("[phase3] Sampling complete!")


def submit_phase3_batch(client, alpha_values, sft_ids, batch_name):
    """Submit Phase 3 retrain jobs for given alpha values (idempotent)."""
    # Check for existing batch records to avoid duplicates
    batch_file = f"artifacts/model_ids/phase3_{batch_name}_jobs.json"
    if os.path.exists(batch_file):
        with open(batch_file) as f:
            existing = json.load(f)
        print(f"[phase3] {batch_file} exists, reusing {len(existing)} job IDs")
        return {jid: tuple(info) for jid, info in existing.items()}

    submitted = {}
    for alpha in alpha_values:
        alpha_tag = f'{int(alpha*10):02d}' if alpha < 1.0 else '10'
        for cluster in config.CLUSTER_NAMES:
            c_lower = cluster.lower()
            path = f"data/preference/grpo_41mini_alpha{alpha_tag}_{c_lower}.jsonl"
            if not os.path.exists(path):
                print(f"[phase3] {path} not found, skipping")
                continue
            try:
                with open(path, "rb") as f:
                    file_resp = client.files.create(file=f, purpose="fine-tune")
                job = client.fine_tuning.jobs.create(
                    training_file=file_resp.id,
                    model=sft_ids[cluster],
                    hyperparameters={"n_epochs": 1},
                    suffix=f"bpo-grpo41-a{alpha_tag}-{c_lower}",
                )
                print(f"  {cluster} alpha={alpha}: {job.id}")
                submitted[job.id] = (f"grpo_41mini_alpha{alpha_tag}", cluster)
            except Exception as e:
                print(f"  {cluster} alpha={alpha}: FAILED ({str(e)[:60]})")

    # Save batch record for idempotency
    os.makedirs("artifacts/model_ids", exist_ok=True)
    with open(batch_file, "w") as f:
        json.dump({jid: list(info) for jid, info in submitted.items()}, f, indent=2)
    return submitted


def wait_for_all_jobs(client, all_job_ids):
    """Wait for all jobs in all_job_ids (dict: id -> (exp, cluster)) to finish."""
    model_ids_by_exp = {}
    while True:
        all_done = True
        try:
            jobs = client.fine_tuning.jobs.list(limit=50)
            for j in jobs.data:
                if j.id not in all_job_ids:
                    continue
                exp, cluster = all_job_ids[j.id]
                if j.status == "succeeded":
                    if exp not in model_ids_by_exp:
                        model_ids_by_exp[exp] = {}
                    model_ids_by_exp[exp][cluster] = j.fine_tuned_model
                elif j.status == "failed":
                    pass  # skip failed
                else:
                    all_done = False

            done_count = sum(len(m) for m in model_ids_by_exp.values())
            total = len(all_job_ids)
            print(f"[jobs] {done_count}/{total} complete")

            if all_done:
                break
        except Exception as e:
            print(f"[jobs] Retry: {str(e)[:60]}")
        time.sleep(120)

    # Save model IDs per experiment
    for exp, models in model_ids_by_exp.items():
        if len(models) == 3:
            path = f"artifacts/model_ids/{exp}_model_ids.json"
            with open(path, "w") as f:
                json.dump(models, f, indent=2)
            print(f"[save] {path}")

    return model_ids_by_exp


def evaluate_experiments(exp_names):
    """Run evaluation on all completed experiments."""
    from models.agent import MultiAgentSystem
    from evaluation.metrics import compute_all_metrics
    import config

    with open("data/splits/test_set.json") as f:
        test_data = json.load(f)

    for exp in exp_names:
        ids_path = f"artifacts/model_ids/{exp}_model_ids.json"
        result_path = f"artifacts/results/{exp}_results.json"

        if not os.path.exists(ids_path):
            print(f"[eval] {exp}: no IDs, skip")
            continue
        if os.path.exists(result_path):
            print(f"[eval] {exp}: already done, skip")
            continue

        with open(ids_path) as f:
            ids = json.load(f)

        print(f"[eval] Running {exp}...")
        system = MultiAgentSystem(model_ids=ids)
        preds = system.predict_batch([t["tweet"] for t in test_data], delay=0.1)
        metrics = compute_all_metrics(preds, test_data)
        metrics["label"] = exp

        os.makedirs("artifacts/results", exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"  MVA={metrics['mva']:.4f}, Disagree={metrics['disagreement_rate']}")


def main():
    client = get_client()

    with open("artifacts/model_ids/sft_41mini_model_ids.json") as f:
        sft_ids = json.load(f)

    # --- Phase 4 jobs (already submitted; just track them) ---
    with open("artifacts/model_ids/phase4_jobs.json") as f:
        phase4_jobs = json.load(f)
    phase4_tracked = {jid: tuple(info) for jid, info in phase4_jobs.items()}

    # --- Wait for Phase 3 sampling to complete ---
    print("\n=== Step 1: Wait for Phase 3 sampling ===")
    wait_for_phase3_sampling()

    # --- Wait for queue slots, submit Phase 3 batch 1 (alpha=0.0, 0.5) ---
    print("\n=== Step 2: Submit Phase 3 batch 1 (alpha=0.0, 0.5) ===")
    if not os.path.exists("artifacts/model_ids/phase3_batch1_jobs.json"):
        wait_for_slots(client, needed=6)
    batch1 = submit_phase3_batch(client, [0.0, 0.5], sft_ids, "batch1")

    # --- Wait for Phase 4 + Phase 3 batch 1 to complete ---
    print("\n=== Step 3: Wait for Phase 4 + Phase 3 batch 1 ===")
    all_tracked = {**phase4_tracked, **batch1}
    wait_for_all_jobs(client, all_tracked)

    # --- Evaluate Phase 4 and Phase 3 batch 1 ---
    print("\n=== Step 4: Evaluate Phase 4 + Phase 3 batch 1 ===")
    phase4_exps = sorted(set(info[0] for info in phase4_tracked.values()))
    batch1_exps = sorted(set(info[0] for info in batch1.values()))
    evaluate_experiments(phase4_exps + batch1_exps)

    # --- Submit Phase 3 batch 2 (alpha=1.0) ---
    print("\n=== Step 5: Submit Phase 3 batch 2 (alpha=1.0) ===")
    if not os.path.exists("artifacts/model_ids/phase3_batch2_jobs.json"):
        wait_for_slots(client, needed=3)
    batch2 = submit_phase3_batch(client, [1.0], sft_ids, "batch2")

    # --- Wait for batch 2 ---
    print("\n=== Step 6: Wait for Phase 3 batch 2 ===")
    wait_for_all_jobs(client, batch2)

    # --- Evaluate Phase 3 batch 2 ---
    print("\n=== Step 7: Evaluate Phase 3 batch 2 ===")
    batch2_exps = sorted(set(info[0] for info in batch2.values()))
    evaluate_experiments(batch2_exps)

    print("\n=== ALL PHASES COMPLETE ===")


if __name__ == "__main__":
    main()
