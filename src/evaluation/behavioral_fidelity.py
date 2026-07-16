#!/usr/bin/env python3
"""
Behavioral fidelity: do the trained agents reproduce the three behavioral
features that defined the annotator clusters?

For each agent (cluster x stage x language), compute from its saved test-set
predictions:
  yes_rate        fraction of YES over all test texts
  agreement_rate  agreement with the human overall majority per text
                  (tie texts excluded) -- mirrors the annotator feature,
                  which measures agreement with the all-annotator majority
  label_entropy   binary Shannon entropy of the agent's YES proportion

Comparison against the human cluster (a distribution over its annotators):
  1. mean/std/CI of the human distribution; agent point + bootstrap 95% CI
     over test texts; z-score and percentile of the agent in the human dist.
  2. K=20 pseudo-annotators drawn from the agent's stored p(YES) values
     (Bernoulli per text), each yielding the 3 features; two-sample
     Wasserstein distance (primary) and KS test (secondary, low power at
     n=20) against the human per-annotator values.

Caveat carried into the paper: human features are computed on each
annotator's own ~57 labeled texts across the whole corpus, agent features on
the test texts only; the group split is random so the two are comparable in
expectation.

Run under .venv-mlx (needs scipy + pandas).
"""
import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

FEATURES = ["yes_rate", "agreement_rate", "label_entropy"]
N_PSEUDO = 20
N_BOOT = 1000


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def agent_features(labels, golds):
    """The 3 behavioral features from one label sequence vs overall majority."""
    yes_rate = sum(1 for l in labels if l == "YES") / len(labels)
    pairs = [(l, g) for l, g in zip(labels, golds) if g is not None]
    agreement = (sum(1 for l, g in pairs if l == g) / len(pairs)) if pairs else None
    return {"yes_rate": yes_rate,
            "agreement_rate": agreement,
            "label_entropy": binary_entropy(yes_rate)}


def bootstrap_ci(labels, golds, feature, n_boot=N_BOOT, seed=config.RANDOM_STATE):
    rng = random.Random(seed)
    n = len(labels)
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        f = agent_features([labels[i] for i in idx], [golds[i] for i in idx])
        if f[feature] is not None:
            vals.append(f[feature])
    vals.sort()
    return [round(vals[int(0.025 * len(vals))], 4),
            round(vals[int(0.975 * len(vals))], 4)]


def pseudo_annotators(p_yes, golds, k=N_PSEUDO, seed=config.RANDOM_STATE):
    """K synthetic annotators sampled from the agent's per-text p(YES)."""
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        labels = ["YES" if rng.random() < p else "NO" for p in p_yes]
        out.append(agent_features(labels, golds))
    return out


def compare(lang, stage_labels):
    behav = pd.read_csv(config.DATA_DIR / f"behavioral_features_{lang}.csv")
    with open(config.SPLITS_DIR / lang / "test_set.json") as f:
        records = json.load(f)
    golds = [r.get("overall_majority") for r in records]

    report = {}
    for label in stage_labels:
        pred_path = config.PREDICTIONS_DIR / f"{label}_predictions.json"
        if not pred_path.exists():
            print(f"[fidelity] missing predictions for {label}, skipping")
            continue
        with open(pred_path) as f:
            preds = json.load(f)
        entry = {}
        for ck in config.CLUSTER_KEYS:
            human = behav[behav["cluster"] == ck]
            labels = [p["agent_predictions"][ck] for p in preds]
            p_yes = [p["agent_p_yes"][ck] for p in preds]
            feats = agent_features(labels, golds)
            pseudo = pseudo_annotators(p_yes, golds)

            block = {"human": {"n": int(len(human))}, "agent": {},
                     "position": {}, "pseudo_annotators": {"k": N_PSEUDO}}
            for feat in FEATURES:
                hvals = human[feat].to_numpy()
                block["human"][feat] = {
                    "mean": round(float(hvals.mean()), 4),
                    "std": round(float(hvals.std(ddof=1)), 4),
                    "ci95": [round(float(np.percentile(hvals, 2.5)), 4),
                             round(float(np.percentile(hvals, 97.5)), 4)],
                }
                av = feats[feat]
                block["agent"][feat] = {
                    "value": round(av, 4) if av is not None else None,
                    "boot_ci95": bootstrap_ci(labels, golds, feat),
                }
                if av is not None and hvals.std(ddof=1) > 0:
                    z = (av - hvals.mean()) / hvals.std(ddof=1)
                    pct = float((hvals < av).mean())
                    block["position"][feat] = {"z": round(float(z), 3),
                                               "percentile": round(pct, 3)}
                pvals = np.array([p[feat] for p in pseudo
                                  if p[feat] is not None])
                if len(pvals):
                    ks = stats.ks_2samp(pvals, hvals)
                    block["pseudo_annotators"][feat] = {
                        "wasserstein": round(float(
                            stats.wasserstein_distance(pvals, hvals)), 4),
                        "ks_stat": round(float(ks.statistic), 4),
                        "ks_p": round(float(ks.pvalue), 4),
                    }
            entry[ck] = block
        report[label] = entry

    out_path = config.RESULTS_DIR / f"behavioral_fidelity_{lang}.json"
    merged = {}
    if out_path.exists():
        with open(out_path) as f:
            merged = json.load(f)
    merged.update(report)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"[fidelity] wrote {out_path} (+{len(report)} models, "
          f"{len(merged)} total)")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--stages", nargs="*",
                    default=["base", "sft", "dpo", "grpo_a20"])
    args = ap.parse_args()
    labels = [f"local_{config.LOCAL_MODEL_LABEL}_{s}_{args.lang}"
              for s in args.stages]
    compare(args.lang, labels)


if __name__ == "__main__":
    main()
