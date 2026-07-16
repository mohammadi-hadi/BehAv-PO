#!/usr/bin/env python3
"""
Phase 4: Multi-Round Deliberation (Optional Extension)
========================================================
Inspired by DMPO's multi-turn formulation.
Agents predict independently, share predictions, then optionally revise.

Algorithm:
  Round 1: Each agent predicts independently
  Round 2: Agents see others' predictions and can revise
  Score: preferred trajectories = reached correct consensus
  Fine-tune on multi-turn preference pairs

Usage:
    python training/deliberation.py --generate       # generate deliberation data
    python training/deliberation.py --finetune       # fine-tune on deliberation data
    python training/deliberation.py --evaluate       # evaluate deliberation
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.agent import MultiAgentSystem

MODELS_DIR = "artifacts/model_ids"
os.makedirs(MODELS_DIR, exist_ok=True)

DELIBERATION_SYSTEM_PROMPT = (
    "You are a content annotator participating in a group review. "
    "Classify whether the given tweet contains sexism. "
    "Respond with exactly YES or NO."
)

REVISION_TEMPLATE = (
    "Tweet: {tweet}\n\n"
    "Your initial assessment: {my_label}\n"
    "Other annotators' assessments: {other_labels}\n\n"
    "After considering the other annotators' views, do you want to change your assessment? "
    "Respond with exactly YES or NO for your final classification."
)


def load_model_ids(source="sft"):
    """Load model IDs for deliberation."""
    if source == "dpo":
        path = os.path.join(MODELS_DIR, "dpo_model_ids.json")
    else:
        path = os.path.join(MODELS_DIR, "sft_model_ids.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as f:
        return json.load(f)


def load_data(split="train"):
    """Load train or test data."""
    path = str(config.SPLITS_DIR / f"{split}_set.json")
    with open(path) as f:
        return json.load(f)


def run_deliberation(system, tweet, client, model_ids):
    """
    Run a 2-round deliberation on a single tweet.

    Returns:
        dict with round1_preds, round2_preds, team_vote_r1, team_vote_r2
    """
    # Round 1: Independent predictions
    r1 = system.predict(tweet, temperature=0.0)
    r1_preds = r1["agent_predictions"]

    # Round 2: Revision with knowledge of others' predictions
    r2_preds = {}
    for cluster in config.CLUSTER_NAMES:
        other_labels = ", ".join(
            f"{c}={r1_preds[c]}" for c in config.CLUSTER_NAMES if c != cluster
        )
        revision_prompt = REVISION_TEMPLATE.format(
            tweet=tweet,
            my_label=r1_preds[cluster],
            other_labels=other_labels,
        )

        response = client.chat.completions.create(
            model=model_ids[cluster],
            messages=[
                {"role": "system", "content": DELIBERATION_SYSTEM_PROMPT},
                {"role": "user", "content": revision_prompt},
            ],
            temperature=0.0,
            max_tokens=3,
        )
        raw = response.choices[0].message.content.strip().upper()
        r2_preds[cluster] = "YES" if "YES" in raw else "NO"

    # Team votes
    r1_yes = sum(1 for v in r1_preds.values() if v == "YES")
    r2_yes = sum(1 for v in r2_preds.values() if v == "YES")
    team_r1 = "YES" if r1_yes >= 2 else "NO"
    team_r2 = "YES" if r2_yes >= 2 else "NO"

    return {
        "round1_preds": r1_preds,
        "round2_preds": r2_preds,
        "team_vote_r1": team_r1,
        "team_vote_r2": team_r2,
    }


def generate_deliberation_data(model_ids, data, output_path, max_texts=500):
    """Generate deliberation trajectories and save."""
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    system = MultiAgentSystem(model_ids=model_ids)

    results = []
    for i, record in enumerate(data[:max_texts]):
        if (i + 1) % 50 == 0:
            print(f"  Processing {i+1}/{min(len(data), max_texts)}...")

        delib = run_deliberation(system, record["tweet"], client, model_ids)
        delib["text_id"] = record["text_id"]
        delib["tweet"] = record["tweet"]
        delib["overall_majority"] = record["overall_majority"]
        delib["cluster_votes"] = record["cluster_votes"]
        results.append(delib)
        time.sleep(0.1)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(results)} deliberation trajectories to {output_path}")

    # Stats
    improved = sum(
        1 for r in results
        if r["overall_majority"] is not None
        and r["team_vote_r2"] == r["overall_majority"]
        and r["team_vote_r1"] != r["overall_majority"]
    )
    degraded = sum(
        1 for r in results
        if r["overall_majority"] is not None
        and r["team_vote_r1"] == r["overall_majority"]
        and r["team_vote_r2"] != r["overall_majority"]
    )
    changed = sum(1 for r in results if r["round1_preds"] != r["round2_preds"])

    print(f"  Agents changed mind: {changed}/{len(results)} texts")
    print(f"  Team improved: {improved}, degraded: {degraded}")

    return results


def build_deliberation_preference_pairs(trajectories, output_path):
    """
    Build multi-turn preference pairs from deliberation trajectories.
    Preferred = trajectories that reached correct consensus.
    """
    pairs = []
    for traj in trajectories:
        if traj["overall_majority"] is None:
            continue

        for cluster in config.CLUSTER_NAMES:
            r1_label = traj["round1_preds"][cluster]
            r2_label = traj["round2_preds"][cluster]

            # Did revision improve alignment with overall majority?
            r1_correct = r1_label == traj["overall_majority"]
            r2_correct = r2_label == traj["overall_majority"]

            if r2_correct and not r1_correct:
                # Good revision: prefer the multi-turn trajectory
                other_labels = ", ".join(
                    f"{c}={traj['round1_preds'][c]}"
                    for c in config.CLUSTER_NAMES if c != cluster
                )
                pair = {
                    "input": [
                        {"role": "system", "content": DELIBERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": REVISION_TEMPLATE.format(
                            tweet=traj["tweet"],
                            my_label=r1_label,
                            other_labels=other_labels,
                        )},
                    ],
                    "preferred_output": [{"role": "assistant", "content": r2_label}],
                    "non_preferred_output": [{"role": "assistant", "content": r1_label}],
                    "cluster": cluster,
                }
                pairs.append(pair)

    # Save
    with open(output_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"  Built {len(pairs)} deliberation preference pairs -> {output_path}")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Multi-round deliberation for BehAv-PO")
    parser.add_argument("--generate", action="store_true", help="Generate deliberation data")
    parser.add_argument("--finetune", action="store_true", help="Fine-tune on deliberation data")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate deliberation")
    parser.add_argument("--source", default="sft", choices=["sft", "dpo"],
                        help="Source models for deliberation")
    parser.add_argument("--max-texts", type=int, default=500)
    args = parser.parse_args()

    model_ids = load_model_ids(args.source)

    if args.generate:
        print("Generating deliberation trajectories...")
        train_data = load_data("train")
        output_path = str(config.PREFERENCE_DIR / "deliberation_trajectories.json")
        trajectories = generate_deliberation_data(
            model_ids, train_data, output_path, max_texts=args.max_texts
        )

        # Build preference pairs
        pairs_path = str(config.PREFERENCE_DIR / "deliberation_pairs.jsonl")
        build_deliberation_preference_pairs(trajectories, pairs_path)

    if args.evaluate:
        print("Evaluating deliberation on test set...")
        test_data = load_data("test")
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        system = MultiAgentSystem(model_ids=model_ids)

        correct_r1, correct_r2, total = 0, 0, 0
        for i, record in enumerate(test_data):
            if record["overall_majority"] is None:
                continue
            if (i + 1) % 50 == 0:
                print(f"  Processing {i+1}/{len(test_data)}...")

            delib = run_deliberation(system, record["tweet"], client, model_ids)
            total += 1
            if delib["team_vote_r1"] == record["overall_majority"]:
                correct_r1 += 1
            if delib["team_vote_r2"] == record["overall_majority"]:
                correct_r2 += 1
            time.sleep(0.1)

        print(f"\n  Round 1 MVA: {correct_r1/total:.4f}")
        print(f"  Round 2 MVA: {correct_r2/total:.4f}")
        print(f"  Improvement: {(correct_r2 - correct_r1)/total:+.4f}")

    if args.finetune:
        print("Fine-tuning on deliberation pairs...")
        pairs_path = str(config.PREFERENCE_DIR / "deliberation_pairs.jsonl")
        if not os.path.exists(pairs_path):
            print(f"  {pairs_path} not found. Run --generate first.")
            return

        client = OpenAI(api_key=config.OPENAI_API_KEY)

        # Upload and fine-tune per cluster
        with open(pairs_path, "rb") as f:
            file_response = client.files.create(file=f, purpose="fine-tune")

        for cluster in config.CLUSTER_NAMES:
            c_lower = cluster.lower()
            try:
                job = client.fine_tuning.jobs.create(
                    training_file=file_response.id,
                    model=model_ids[cluster],
                    method={"type": "dpo", "dpo": {"hyperparameters": {"beta": 0.15}}},
                    suffix=f"behav-po-delib-{c_lower}",
                )
                print(f"  {cluster}: Job {job.id} ({job.status})")
            except Exception as e:
                print(f"  {cluster}: DPO not available, skipping ({e})")


if __name__ == "__main__":
    main()
