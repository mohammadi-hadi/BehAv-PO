#!/usr/bin/env python3
"""
Resume Pipeline After Sleep
============================
Run this when you wake up your computer. It will:

1. Check all OpenAI jobs (Phase 2a + 2b + 2c + any others)
2. Save fine-tuned model IDs for completed jobs
3. Run evaluations on all new models
4. Regenerate plots and paper
5. Commit and push everything

If jobs are still running, it waits for them and retries.

Usage:
    export OPENAI_API_KEY="your-key"
    python3 resume_pipeline.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_and_save_completed_jobs():
    """Check all job files and save model IDs for succeeded ones."""
    from openai import OpenAI
    client = OpenAI()

    print("Checking all fine-tuning jobs...")
    jobs = client.fine_tuning.jobs.list(limit=50)
    jobs_by_id = {j.id: j for j in jobs.data}

    # Map suffixes to experiment types
    experiments = {
        "behav-po-sftpure": "sft_pure_41mini",
        "behav-po-marspo": "marspo_41mini",
        "behav-po-grpo41-a02": "grpo_41mini_alpha02",
    }

    results = {}
    for job in jobs.data:
        if job.status != "succeeded":
            continue
        suffix = getattr(job, "user_provided_suffix", None) or ""
        model_name = job.fine_tuned_model or ""

        # Match experiment
        for prefix, exp_name in experiments.items():
            if prefix in suffix or prefix in model_name:
                # Extract cluster name
                for cluster in ["conservative", "mainstream", "sensitive"]:
                    if cluster in suffix.lower() or cluster in model_name.lower():
                        if exp_name not in results:
                            results[exp_name] = {}
                        results[exp_name][cluster.capitalize()] = model_name
                        break
                break

    # Save to disk
    for exp_name, models in results.items():
        if len(models) == 3:
            path = f"artifacts/model_ids/{exp_name}_model_ids.json"
            with open(path, "w") as f:
                json.dump(models, f, indent=2)
            print(f"  Saved {path}: {list(models.keys())}")
        else:
            print(f"  {exp_name}: only {len(models)}/3 models ready, skipping")

    return results


def evaluate_all_new_experiments():
    """Run evaluation on all new experiment models."""
    from models.agent import MultiAgentSystem
    from evaluation.metrics import compute_all_metrics
    import config

    with open("data/splits/test_set.json") as f:
        test_data = json.load(f)

    new_experiments = [
        "sft_pure_41mini",
        "marspo_41mini",
        "grpo_41mini_alpha02",
    ]

    for exp in new_experiments:
        ids_path = f"artifacts/model_ids/{exp}_model_ids.json"
        result_path = f"artifacts/results/{exp}_results.json"

        if not os.path.exists(ids_path):
            print(f"  {exp}: model IDs not found, skipping")
            continue
        if os.path.exists(result_path):
            print(f"  {exp}: already evaluated, skipping")
            continue

        with open(ids_path) as f:
            model_ids = json.load(f)

        print(f"\n  Evaluating {exp}...")
        system = MultiAgentSystem(model_ids=model_ids)
        preds = system.predict_batch([t["tweet"] for t in test_data], delay=0.1)
        metrics = compute_all_metrics(preds, test_data)
        metrics["label"] = exp

        os.makedirs("artifacts/results", exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"    MVA={metrics['mva']:.4f}, Disagree={metrics['disagreement_rate']}")


def maybe_submit_phase3(dry_run=False):
    """
    Submit Phase 3 (alpha sweep) jobs if queue has slots.

    alpha = 0.0 (individual only)
    alpha = 0.5 (balanced)
    alpha = 1.0 (team heavy)
    """
    from openai import OpenAI
    client = OpenAI()

    # Check we have GRPO alpha=0.2 done first
    if not os.path.exists("artifacts/model_ids/grpo_41mini_alpha02_model_ids.json"):
        print("Phase 2c (alpha=0.2) not done yet, skipping Phase 3")
        return

    # Check queue
    jobs = client.fine_tuning.jobs.list(limit=30)
    active = sum(
        1 for j in jobs.data
        if j.status in ["queued", "validating_files", "running"]
        and "gpt-4.1-mini" in (j.model or "")
    )
    slots_free = 6 - active
    print(f"  Queue: {active}/6 active, {slots_free} free")

    if slots_free < 3:
        print("  Not enough slots for 3 alpha sweep jobs, retry later")
        return

    # TODO: implement alpha sweep submission here
    print("  Phase 3 submission: not yet implemented")


def main():
    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: Set OPENAI_API_KEY env var first")
        sys.exit(1)

    print("=" * 60)
    print("Pipeline Resume")
    print("=" * 60)

    # 1. Check and save completed jobs
    print("\nStep 1: Check completed OpenAI jobs")
    results = check_and_save_completed_jobs()

    # 2. Wait for any pending jobs (up to 30 min)
    print("\nStep 2: Wait for any remaining jobs (up to 30 min)")
    from openai import OpenAI
    client = OpenAI()
    max_wait_min = 30
    for i in range(max_wait_min // 2):
        jobs = client.fine_tuning.jobs.list(limit=30)
        active = [j for j in jobs.data
                  if j.status in ["queued", "validating_files", "running"]]
        if not active:
            print("  All jobs done")
            break
        print(f"  {len(active)} jobs still active, waiting 2 min...")
        time.sleep(120)
        # Recheck and save
        results = check_and_save_completed_jobs()

    # 3. Evaluate new experiments
    print("\nStep 3: Evaluate new experiments")
    evaluate_all_new_experiments()

    # 4. Regenerate plots
    print("\nStep 4: Regenerate plots")
    os.system("python3 src/evaluation/generate_plots.py")

    # 5. Run analysis
    print("\nStep 5: Run statistical analysis")
    os.system("python3 src/evaluation/analysis.py")

    # 6. Verify pipeline
    print("\nStep 6: Verify pipeline")
    exit_code = os.system("python3 src/pipeline/verify_pipeline.py")

    if exit_code == 0:
        print("\nAll checks passed! Ready to commit and push.")
        print("Run:")
        print('  git add <files>')
        print('  git -c user.name="mohammadi-hadi" \\')
        print('       -c user.email="50410241+mohammadi-hadi@users.noreply.github.com" \\')
        print('       commit -m "..."')
        print("  git push origin main")
    else:
        print("\nVerifier found issues, review before committing.")


if __name__ == "__main__":
    main()
