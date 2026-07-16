#!/usr/bin/env python3
"""
Evaluation Pipeline
====================
Evaluates the BehAv-PO multi-agent system on the test set.
Supports evaluation of: base model, SFT models, DPO models, GRPO models.
Includes baseline comparisons.

Usage:
    python evaluation/evaluate_agents.py --model sft       # evaluate SFT models
    python evaluation/evaluate_agents.py --model dpo       # evaluate DPO models
    python evaluation/evaluate_agents.py --model grpo      # evaluate GRPO models
    python evaluation/evaluate_agents.py --model base      # evaluate base model (zero-shot)
    python evaluation/evaluate_agents.py --baselines        # run all baselines
"""

import argparse
import json
import os
import sys
import time

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.agent import Agent, MultiAgentSystem
from evaluation.metrics import compute_all_metrics

MODELS_DIR = "artifacts/model_ids"
RESULTS_DIR = "artifacts/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data():
    """Load test set."""
    path = str(config.SPLITS_DIR / "test_set.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run data/prepare_preference_data.py first."
        )
    with open(path) as f:
        return json.load(f)


def load_model_ids(model_type, iteration=None):
    """Load model IDs for a given phase."""
    if model_type == "base":
        return {c: config.BASE_MODEL for c in config.CLUSTER_NAMES}
    elif model_type == "sft":
        path = os.path.join(MODELS_DIR, "sft_model_ids.json")
    elif model_type == "dpo":
        path = os.path.join(MODELS_DIR, "dpo_model_ids.json")
    elif model_type == "grpo":
        it = iteration or config.GRPO_ITERATIONS
        path = os.path.join(MODELS_DIR, f"grpo_iter{it}_model_ids.json")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Complete {model_type} training first.")
    with open(path) as f:
        return json.load(f)


def evaluate_multi_agent(model_ids, test_data, label=""):
    """Run multi-agent evaluation and return metrics."""
    print(f"\nEvaluating: {label}")
    print(f"  Models: {model_ids}")
    print(f"  Test texts: {len(test_data)}")

    system = MultiAgentSystem(model_ids=model_ids)
    tweets = [t["tweet"] for t in test_data]

    print("  Running inference...")
    results = system.predict_batch(tweets, delay=0.05)

    metrics = compute_all_metrics(results, test_data)
    metrics["label"] = label
    metrics["n_texts"] = len(test_data)

    print(f"\n  Results ({label}):")
    print(f"    MVA (team accuracy):       {metrics.get('mva', 'N/A')}")
    print(f"    F1-macro:                  {metrics.get('f1_macro', 'N/A')}")
    print(f"    Disagreement rate:         {metrics.get('disagreement_rate', 'N/A')}")
    for cluster in config.CLUSTER_NAMES:
        c_lower = cluster.lower()
        print(f"    CAA ({cluster}):  {metrics.get(f'caa_{c_lower}', 'N/A')}")
        print(f"    YES rate ({cluster}): {metrics.get(f'yes_rate_{c_lower}', 'N/A')} "
              f"(expected: {config.EXPECTED_YES_RATES.get(cluster, '?')})")

    return metrics, results


def evaluate_baseline_always_no(test_data):
    """Baseline: always predict NO."""
    print("\nBaseline: Always-NO")
    correct = sum(1 for t in test_data if t["overall_majority"] == "NO")
    total = sum(1 for t in test_data if t["overall_majority"] is not None)
    acc = correct / total if total > 0 else 0
    print(f"  Accuracy: {acc:.4f} ({correct}/{total})")
    return {"label": "always_no", "mva": round(acc, 4), "n_texts": len(test_data)}


def evaluate_baseline_single_agent(test_data):
    """Baseline: single GPT-4o-mini on overall majority (no cluster conditioning)."""
    print("\nBaseline: Single Agent (zero-shot)")
    agent = Agent("single", config.BASE_MODEL)
    tweets = [t["tweet"] for t in test_data]

    results = agent.predict_batch(tweets, delay=0.05)
    predictions = [r["label"] for r in results]

    correct = 0
    total = 0
    for pred, t in zip(predictions, test_data):
        if t["overall_majority"] is not None:
            total += 1
            if pred == t["overall_majority"]:
                correct += 1

    acc = correct / total if total > 0 else 0
    yes_rate = sum(1 for p in predictions if p == "YES") / len(predictions)
    print(f"  Accuracy: {acc:.4f}")
    print(f"  YES rate: {yes_rate:.4f}")
    return {
        "label": "single_agent_zero_shot",
        "mva": round(acc, 4),
        "yes_rate": round(yes_rate, 4),
        "n_texts": len(test_data),
    }


def evaluate_baseline_persona_prompt(test_data):
    """Baseline: single GPT-4o-mini with persona prompts (no fine-tuning)."""
    print("\nBaseline: Persona Prompts (no fine-tuning)")

    persona_prompts = {
        "Conservative": (
            "You are a strict content annotator who rarely flags content as sexist. "
            "You only label clear, explicit cases of sexism. "
            "Classify whether the given tweet contains sexism. "
            "Respond with exactly YES or NO."
        ),
        "Mainstream": (
            "You are a balanced content annotator. "
            "You label content as sexist when it contains clear or moderately implicit sexism. "
            "Classify whether the given tweet contains sexism. "
            "Respond with exactly YES or NO."
        ),
        "Sensitive": (
            "You are a sensitive content annotator who is attuned to subtle forms of sexism. "
            "You flag content as sexist even when the sexism is implicit or borderline. "
            "Classify whether the given tweet contains sexism. "
            "Respond with exactly YES or NO."
        ),
    }

    client = OpenAI(api_key=config.OPENAI_API_KEY)
    all_preds = {}

    for cluster, prompt in persona_prompts.items():
        agent = Agent(cluster)
        preds = []
        for t in test_data:
            response = client.chat.completions.create(
                model=config.BASE_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Tweet: {t['tweet']}"},
                ],
                temperature=0.0,
                max_tokens=3,
            )
            raw = response.choices[0].message.content.strip().upper()
            label = "YES" if "YES" in raw else "NO"
            preds.append(label)
            time.sleep(0.05)
        all_preds[cluster] = preds
        yes_rate = sum(1 for p in preds if p == "YES") / len(preds)
        print(f"  {cluster} YES rate: {yes_rate:.4f}")

    # Compute team vote
    correct, total = 0, 0
    for i, t in enumerate(test_data):
        if t["overall_majority"] is None:
            continue
        votes = [all_preds[c][i] for c in config.CLUSTER_NAMES]
        yes_count = sum(1 for v in votes if v == "YES")
        team_vote = "YES" if yes_count >= 2 else "NO"
        total += 1
        if team_vote == t["overall_majority"]:
            correct += 1

    acc = correct / total if total > 0 else 0
    print(f"  Team MVA: {acc:.4f}")
    return {
        "label": "persona_prompt",
        "mva": round(acc, 4),
        "n_texts": len(test_data),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate BehAv-PO agents")
    parser.add_argument("--model", choices=["base", "sft", "dpo", "grpo"],
                        help="Model phase to evaluate")
    parser.add_argument("--baselines", action="store_true", help="Run all baselines")
    parser.add_argument("--all", action="store_true", help="Run all evaluations")
    parser.add_argument("--grpo-iter", type=int, help="GRPO iteration to evaluate")
    args = parser.parse_args()

    test_data = load_test_data()
    all_results = []

    if args.baselines or args.all:
        # Always-NO baseline
        r = evaluate_baseline_always_no(test_data)
        all_results.append(r)

        # Single agent zero-shot
        r = evaluate_baseline_single_agent(test_data)
        all_results.append(r)

        # Persona prompt
        r = evaluate_baseline_persona_prompt(test_data)
        all_results.append(r)

    if args.model or args.all:
        models_to_eval = [args.model] if args.model else ["base", "sft", "dpo", "grpo"]
        for model_type in models_to_eval:
            try:
                model_ids = load_model_ids(model_type, iteration=args.grpo_iter)
                metrics, _ = evaluate_multi_agent(
                    model_ids, test_data, label=f"multi_agent_{model_type}"
                )
                all_results.append(metrics)
            except FileNotFoundError as e:
                print(f"\nSkipping {model_type}: {e}")

    if all_results:
        # Save results
        results_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to {results_path}")

        # Summary table
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"{'Method':<35} {'MVA':>8} {'F1-macro':>10}")
        print("-" * 55)
        for r in all_results:
            mva = r.get("mva", "N/A")
            f1 = r.get("f1_macro", "N/A")
            mva_str = f"{mva:.4f}" if isinstance(mva, float) else str(mva)
            f1_str = f"{f1:.4f}" if isinstance(f1, float) else str(f1)
            print(f"{r['label']:<35} {mva_str:>8} {f1_str:>10}")


if __name__ == "__main__":
    main()
