#!/usr/bin/env python3
"""
Phase 3: GRPO via Rejection Sampling
======================================
Approximates GRPO (Group Relative Policy Optimization) without gradient access.
Implements MAS-LLM's R_j = r_indiv + alpha * r_team through rejection sampling.

Algorithm per iteration:
  1. Sample K completions per text from each agent
  2. Score with individual + team reward
  3. Keep high-reward samples
  4. Re-fine-tune each agent on filtered samples (SFT API)

Usage:
    python training/grpo_rejection_sampling.py               # run all iterations
    python training/grpo_rejection_sampling.py --iteration 1  # run specific iteration
    python training/grpo_rejection_sampling.py --status        # check job statuses
"""

import argparse
import json
import os
import random
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.agent import Agent

MODELS_DIR = "artifacts/model_ids"
os.makedirs(MODELS_DIR, exist_ok=True)


def load_current_model_ids(iteration):
    """Load model IDs for the current iteration."""
    if iteration == 1:
        # Start from SFT models
        path = os.path.join(MODELS_DIR, "sft_model_ids.json")
    else:
        path = os.path.join(MODELS_DIR, f"grpo_iter{iteration-1}_model_ids.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Complete the previous phase first.")
    with open(path) as f:
        return json.load(f)


def load_train_data():
    """Load training set with ground truth labels."""
    path = str(config.SPLITS_DIR / "train_set.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data/prepare_preference_data.py first."
        )
    with open(path) as f:
        return json.load(f)


def compute_reward(agent_label, cluster_name, text_record, all_agent_labels):
    """
    Compute reward for one agent's prediction on one text.

    R_j = r_indiv_j + alpha * r_team

    Args:
        agent_label: "YES" or "NO" predicted by this agent
        cluster_name: which cluster this agent represents
        text_record: dict with cluster_votes, overall_majority
        all_agent_labels: dict {cluster: "YES"/"NO"} from all 3 agents

    Returns:
        float: total reward
    """
    # Individual reward: does prediction match cluster's ground truth?
    cv = text_record.get("cluster_votes", {})
    r_indiv = 0.0
    if cluster_name in cv and cv[cluster_name].get("majority") is not None:
        r_indiv = 1.0 if agent_label == cv[cluster_name]["majority"] else 0.0

    # Team reward: does majority vote match overall majority?
    r_team = 0.0
    if all_agent_labels and text_record.get("overall_majority") is not None:
        yes_count = sum(1 for v in all_agent_labels.values() if v == "YES")
        team_vote = "YES" if yes_count >= 2 else "NO"
        r_team = 1.0 if team_vote == text_record["overall_majority"] else 0.0

    return r_indiv + config.GRPO_ALPHA * r_team


def sample_and_score(model_ids, train_data, sample_size=None):
    """
    Sample K completions per text from each agent and score them.

    Returns:
        dict {cluster: [{tweet, label, reward}, ...]}
    """
    if sample_size:
        random.seed(config.RANDOM_STATE)
        train_data = random.sample(train_data, min(sample_size, len(train_data)))

    agents = {
        cluster: Agent(cluster, model_ids.get(cluster))
        for cluster in config.CLUSTER_NAMES
    }

    all_samples = {cluster: [] for cluster in config.CLUSTER_NAMES}

    for i, record in enumerate(train_data):
        tweet = record["tweet"]
        if (i + 1) % 100 == 0:
            print(f"  Processing text {i+1}/{len(train_data)}...")

        # Sample K completions from each agent
        agent_samples = {}
        for cluster in config.CLUSTER_NAMES:
            samples = agents[cluster].sample_multiple(
                tweet,
                k=config.GRPO_K_SAMPLES,
                temperature=config.GRPO_TEMPERATURE,
            )
            agent_samples[cluster] = [s["label"] for s in samples]

        # Score each sample
        for cluster in config.CLUSTER_NAMES:
            for k_idx in range(config.GRPO_K_SAMPLES):
                # For team reward, use the most common label from other agents
                other_labels = {}
                for other_cluster in config.CLUSTER_NAMES:
                    if other_cluster != cluster:
                        # Use the majority of K samples from other agents
                        other_preds = agent_samples[other_cluster]
                        yes_count = sum(1 for p in other_preds if p == "YES")
                        other_labels[other_cluster] = "YES" if yes_count > len(other_preds) / 2 else "NO"
                other_labels[cluster] = agent_samples[cluster][k_idx]

                reward = compute_reward(
                    agent_samples[cluster][k_idx],
                    cluster,
                    record,
                    other_labels,
                )

                all_samples[cluster].append({
                    "tweet": tweet,
                    "label": agent_samples[cluster][k_idx],
                    "reward": reward,
                })

    return all_samples


def filter_and_build_sft(all_samples, iteration):
    """
    Filter high-reward samples and build SFT datasets for re-training.

    Returns:
        dict {cluster: filepath}
    """
    paths = {}
    for cluster in config.CLUSTER_NAMES:
        c_lower = cluster.lower()
        samples = all_samples[cluster]

        # Filter by reward threshold
        high_reward = [s for s in samples if s["reward"] > config.GRPO_REWARD_THRESHOLD]

        # Deduplicate by (tweet, label) keeping highest reward
        seen = {}
        for s in high_reward:
            key = (s["tweet"], s["label"])
            if key not in seen or s["reward"] > seen[key]["reward"]:
                seen[key] = s
        filtered = list(seen.values())

        # Build SFT JSONL
        sft_data = []
        for s in filtered:
            sft_data.append({
                "messages": [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Tweet: {s['tweet']}"},
                    {"role": "assistant", "content": s["label"]},
                ]
            })

        path = str(config.PREFERENCE_DIR / f"grpo_iter{iteration}_{c_lower}.jsonl")
        with open(path, "w") as f:
            for item in sft_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        paths[cluster] = path
        print(f"  {cluster}: {len(samples)} samples -> {len(high_reward)} above threshold -> "
              f"{len(filtered)} unique -> {path}")

    return paths


def submit_retrain_jobs(client, model_ids, sft_paths, iteration):
    """Submit SFT re-training jobs with filtered GRPO data."""
    jobs = []
    for cluster in config.CLUSTER_NAMES:
        c_lower = cluster.lower()
        base_model = model_ids[cluster]
        train_path = sft_paths[cluster]

        print(f"\n  Submitting GRPO retrain for {cluster} (iter {iteration})...")
        print(f"    Base: {base_model}")

        with open(train_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="fine-tune")

        job = client.fine_tuning.jobs.create(
            training_file=file_response.id,
            model=base_model,
            hyperparameters={"n_epochs": 1},
            suffix=f"behav-po-grpo-iter{iteration}-{c_lower}",
        )
        print(f"    Job ID: {job.id}")

        # Save job info
        job_info = {
            "job_id": job.id,
            "cluster": cluster,
            "base_model": base_model,
            "iteration": iteration,
            "status": job.status,
        }
        with open(os.path.join(MODELS_DIR, f"grpo_iter{iteration}_{c_lower}_job.json"), "w") as f:
            json.dump(job_info, f, indent=2)

        jobs.append(job)

    return jobs


def check_status(client, iteration=None):
    """Check GRPO job statuses."""
    iterations = [iteration] if iteration else range(1, config.GRPO_ITERATIONS + 1)

    for it in iterations:
        print(f"\n{'='*50}")
        print(f"GRPO Iteration {it} Status")
        print(f"{'='*50}")

        model_ids = {}
        for cluster in config.CLUSTER_NAMES:
            c_lower = cluster.lower()
            path = os.path.join(MODELS_DIR, f"grpo_iter{it}_{c_lower}_job.json")
            if not os.path.exists(path):
                print(f"  {cluster}: No job found")
                continue

            with open(path) as f:
                info = json.load(f)

            job = client.fine_tuning.jobs.retrieve(info["job_id"])
            print(f"  {cluster}: {job.status}")
            if job.status == "succeeded":
                print(f"    Model: {job.fine_tuned_model}")
                model_ids[cluster] = job.fine_tuned_model
                info["status"] = "succeeded"
                info["fine_tuned_model"] = job.fine_tuned_model
                with open(path, "w") as f:
                    json.dump(info, f, indent=2)

        if len(model_ids) == len(config.CLUSTER_NAMES):
            ids_path = os.path.join(MODELS_DIR, f"grpo_iter{it}_model_ids.json")
            with open(ids_path, "w") as f:
                json.dump(model_ids, f, indent=2)
            print(f"  All iter {it} models ready! Saved to {ids_path}")


def main():
    parser = argparse.ArgumentParser(description="GRPO rejection sampling for BehAv-PO")
    parser.add_argument("--iteration", type=int, help="Run specific iteration")
    parser.add_argument("--status", action="store_true", help="Check job statuses")
    parser.add_argument("--sample-size", type=int, default=500,
                        help="Number of texts to sample per iteration (default: 500)")
    args = parser.parse_args()

    client = OpenAI(api_key=config.OPENAI_API_KEY)

    if args.status:
        check_status(client, args.iteration)
        return

    iterations = [args.iteration] if args.iteration else range(1, config.GRPO_ITERATIONS + 1)

    for iteration in iterations:
        print(f"\n{'='*60}")
        print(f"GRPO Iteration {iteration}/{config.GRPO_ITERATIONS}")
        print(f"{'='*60}")

        # Load current models
        model_ids = load_current_model_ids(iteration)
        print(f"\nCurrent models:")
        for c, mid in model_ids.items():
            print(f"  {c}: {mid}")

        # Load training data
        train_data = load_train_data()

        # Sample and score
        print(f"\nSampling {config.GRPO_K_SAMPLES} completions per text "
              f"from {args.sample_size} texts...")
        all_samples = sample_and_score(model_ids, train_data, sample_size=args.sample_size)

        # Filter and build SFT data
        print("\nFiltering high-reward samples...")
        sft_paths = filter_and_build_sft(all_samples, iteration)

        # Submit retrain jobs
        print("\nSubmitting retrain jobs...")
        jobs = submit_retrain_jobs(client, model_ids, sft_paths, iteration)
        print(f"\n{len(jobs)} jobs submitted for iteration {iteration}.")
        print("Run with --status to check progress before starting next iteration.")
        break  # Only run one iteration at a time


if __name__ == "__main__":
    main()
