#!/usr/bin/env python3
"""
Statistical Analysis and Error Analysis
========================================
Generates:
- Bootstrap confidence intervals on MVA, F1, CAA per method
- Confusion matrices per method
- Disagreement patterns (which cluster "carries" the tie)
- Per-text error analysis

Usage:
    python evaluation/analysis.py
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

RESULTS_DIR = "artifacts/results"
N_BOOTSTRAP = 1000


def bootstrap_ci(values, n=N_BOOTSTRAP, alpha=0.05):
    """Bootstrap CI on a binary array (1=correct, 0=wrong)."""
    values = np.array(values)
    if len(values) == 0:
        return None, None, None
    means = []
    rng = np.random.default_rng(42)
    for _ in range(n):
        sample = rng.choice(values, size=len(values), replace=True)
        means.append(sample.mean())
    lo = np.quantile(means, alpha / 2)
    hi = np.quantile(means, 1 - alpha / 2)
    return values.mean(), lo, hi


def rerun_predictions_from_saved():
    """
    If we have saved predictions, use them. Otherwise we'll need to
    compute from stored metrics only.
    """
    # For now, we compute CIs on the metrics we have by re-evaluating
    # at test time. Here we load saved predictions if available.
    pass


def analyze_confusion_and_disagreement(predictions_path, method_name):
    """
    Given a file with per-text predictions, analyze confusion patterns.

    Format expected:
      [{text_id, predictions: {cluster: YES/NO}, team_vote, overall_majority, ...}]
    """
    if not os.path.exists(predictions_path):
        return None

    with open(predictions_path) as f:
        preds = json.load(f)

    # Team confusion matrix (vs overall majority)
    cm = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}
    for p in preds:
        team = p["team_vote"]
        gold = p.get("overall_majority")
        if gold is None:
            continue
        if team == "YES" and gold == "YES":
            cm["TP"] += 1
        elif team == "NO" and gold == "NO":
            cm["TN"] += 1
        elif team == "YES" and gold == "NO":
            cm["FP"] += 1
        elif team == "NO" and gold == "YES":
            cm["FN"] += 1

    # Disagreement patterns: who is the "lone dissenter"?
    lone_dissenter = Counter()
    unanimous = 0
    disagree_outcomes = Counter()
    for p in preds:
        agent_preds = p["predictions"]
        values = list(agent_preds.values())
        if len(set(values)) == 1:
            unanimous += 1
            continue
        # Find the dissenter (minority)
        yes_count = sum(1 for v in values if v == "YES")
        if yes_count == 1:
            # 1 YES, 2 NO — YES agent is dissenter
            for name, v in agent_preds.items():
                if v == "YES":
                    lone_dissenter[name] += 1
        elif yes_count == 2:
            for name, v in agent_preds.items():
                if v == "NO":
                    lone_dissenter[name] += 1

        # Outcome: did the team vote match human majority?
        team = p["team_vote"]
        gold = p.get("overall_majority")
        if gold is not None:
            disagree_outcomes[f"team_{team}_gold_{gold}"] += 1

    return {
        "method": method_name,
        "confusion_matrix": cm,
        "precision_yes": cm["TP"] / (cm["TP"] + cm["FP"]) if (cm["TP"] + cm["FP"]) > 0 else 0,
        "recall_yes": cm["TP"] / (cm["TP"] + cm["FN"]) if (cm["TP"] + cm["FN"]) > 0 else 0,
        "precision_no": cm["TN"] / (cm["TN"] + cm["FN"]) if (cm["TN"] + cm["FN"]) > 0 else 0,
        "recall_no": cm["TN"] / (cm["TN"] + cm["FP"]) if (cm["TN"] + cm["FP"]) > 0 else 0,
        "unanimous_count": unanimous,
        "disagreement_count": len(preds) - unanimous,
        "lone_dissenter": dict(lone_dissenter),
        "disagree_outcomes": dict(disagree_outcomes),
    }


def main():
    print("=" * 60)
    print("Statistical Analysis & Error Analysis")
    print("=" * 60)

    results_summary = {}

    # Load all saved results
    result_files = {
        "always_no": "baseline_results.json",
        "single_zero_shot": "baseline_results.json",
        "zero_shot_ensemble": "baseline_results.json",
        "persona_prompts": "baseline_results.json",
        "sft": "sft_results.json",
        "grpo": "grpo_results.json",
        "sft_41mini": "sft_41mini_results.json",
        "sft_pure_41mini": "sft_pure_41mini_results.json",
        "dpo_41mini": "dpo_41mini_results.json",
        "dpo_bb03_41mini": "dpo_bb03_41mini_results.json",
        "dpo_bb05_41mini": "dpo_bb05_41mini_results.json",
        "marspo_41mini": "marspo_41mini_results.json",
        "grpo_41mini_alpha00": "grpo_41mini_alpha00_results.json",
        "grpo_41mini_alpha02": "grpo_41mini_alpha02_results.json",
        "grpo_41mini_alpha05": "grpo_41mini_alpha05_results.json",
        "grpo_41mini_alpha10": "grpo_41mini_alpha10_results.json",
        # Legacy alias so prior code paths still work
        "grpo_41mini": "grpo_41mini_alpha02_results.json",
    }

    # Load all metrics
    all_metrics = {}
    for method, fname in result_files.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        # Handle lists vs dicts
        if isinstance(data, list):
            for entry in data:
                if entry.get("label") == method:
                    all_metrics[method] = entry
                    break
        else:
            # Accept by exact label, startswith, or filename match
            label = data.get("label", "")
            if label == method or label.startswith(method) or method in label:
                all_metrics[method] = data
            elif not all_metrics.get(method):
                # Fallback: file name matches
                all_metrics[method] = data

    print(f"\nLoaded {len(all_metrics)} methods:")
    for m in all_metrics:
        print(f"  {m}: MVA={all_metrics[m].get('mva', 'N/A')}")

    # Bootstrap CIs on MVA (requires test set size)
    print("\n" + "=" * 60)
    print("Bootstrap Confidence Intervals (n={})".format(N_BOOTSTRAP))
    print("=" * 60)

    with open("data/splits/test_set.json") as f:
        test_data = json.load(f)
    n_with_majority = sum(1 for t in test_data if t.get("overall_majority") is not None)
    print(f"Test texts with clear majority: {n_with_majority}")

    ci_results = {}
    for method, m in all_metrics.items():
        mva = m.get("mva")
        if mva is None:
            continue

        # Simulate binary outcomes based on reported MVA
        # This gives us a bootstrap CI assuming we know the rate
        n_correct = int(round(mva * n_with_majority))
        binary = [1] * n_correct + [0] * (n_with_majority - n_correct)

        mean, lo, hi = bootstrap_ci(binary, n=N_BOOTSTRAP)
        ci_results[method] = {
            "mva": round(mean, 4),
            "ci_lo": round(lo, 4),
            "ci_hi": round(hi, 4),
            "ci_width": round(hi - lo, 4),
        }
        print(f"  {method:25s} MVA={mean:.3f} [{lo:.3f}, {hi:.3f}]")

    # Significance test: is BehAv-PO significantly better than baselines?
    print("\n" + "=" * 60)
    print("Pairwise Significance (95% CI overlap)")
    print("=" * 60)

    baselines = ["always_no", "single_zero_shot", "zero_shot_ensemble", "persona_prompts"]
    ours = [
        "sft", "grpo",
        "sft_41mini", "sft_pure_41mini",
        "dpo_41mini", "dpo_bb03_41mini", "dpo_bb05_41mini",
        "marspo_41mini",
        "grpo_41mini_alpha00", "grpo_41mini_alpha02",
        "grpo_41mini_alpha05", "grpo_41mini_alpha10",
    ]

    sig_results = []
    for b in baselines:
        if b not in ci_results:
            continue
        for o in ours:
            if o not in ci_results:
                continue
            b_ci = ci_results[b]
            o_ci = ci_results[o]
            # Non-overlapping CIs => p<0.05 roughly
            sig = bool(o_ci["ci_lo"] > b_ci["ci_hi"])
            diff = float(o_ci["mva"] - b_ci["mva"])
            sig_results.append({
                "method": o, "baseline": b, "diff": round(diff, 4),
                "significant": sig,
            })
            marker = "**" if sig else ""
            print(f"  {o:25s} vs {b:25s} diff={diff:+.4f} {marker}")

    # Save everything
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "confidence_intervals.json"), "w") as f:
        json.dump(ci_results, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "significance_tests.json"), "w") as f:
        json.dump(sig_results, f, indent=2)

    # Error analysis from saved SFT predictions
    print("\n" + "=" * 60)
    print("Error Analysis (SFT predictions)")
    print("=" * 60)

    ea = analyze_confusion_and_disagreement(
        os.path.join(RESULTS_DIR, "sft_predictions.json"), "sft"
    )
    if ea:
        print(f"  Confusion: {ea['confusion_matrix']}")
        print(f"  Precision YES: {ea['precision_yes']:.3f}")
        print(f"  Recall YES:    {ea['recall_yes']:.3f}")
        print(f"  Precision NO:  {ea['precision_no']:.3f}")
        print(f"  Recall NO:     {ea['recall_no']:.3f}")
        print(f"  Unanimous: {ea['unanimous_count']}")
        print(f"  Disagreement: {ea['disagreement_count']}")
        print(f"  Lone dissenter counts: {ea['lone_dissenter']}")
        with open(os.path.join(RESULTS_DIR, "error_analysis_sft.json"), "w") as f:
            json.dump(ea, f, indent=2)

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
