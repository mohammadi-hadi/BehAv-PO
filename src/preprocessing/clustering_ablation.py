#!/usr/bin/env python3
"""
Clustering Feature Ablation: drop yes_rate
===========================================
Coauthor request: evaluate KMeans clustering of annotators using ONLY
(agreement_rate, label_entropy) -- i.e. without yes_rate -- and compare
against the primary 3-feature clustering (yes_rate, agreement_rate,
label_entropy) stored in data/behavioral_features_{lang}.csv.

Reports, per language, at the silhouette-best k (scan 2..10) AND at k=3:
  cluster sizes, min share, normalized size entropy, silhouette,
  ARI + crosstab vs the 3-feature clustering,
  per-cluster mean yes_rate + min pairwise |gap| (yes-rate separation).

Decision rule (encoded in the "recommendation" field, evaluated at k=3
because the multi-agent pipeline uses per-cluster agents): adopt only if
  min share >= 0.15
  AND min share > 3-feature min share
  AND silhouette >= (3-feature silhouette - 0.05)
  AND all pairwise mean-yes-rate gaps >= 0.10
else "report-only".

Output: artifacts/results/clustering_ablation_{lang}.json

Usage:
  python3 src/preprocessing/clustering_ablation.py --lang en|es
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from behavioral_clustering import (
    build_label_matrix,
    compute_behavioral_features,
    scan_kmeans,
    fit_kmeans,
    name_clusters_by_yes_rate,
)

ABLATION_FEATURES = ["agreement_rate", "label_entropy"]
K_RANGE = range(2, 11)

KEYWORD_FEATURES_NOTE = (
    "Text-content (keyword) features were NOT used for annotator clustering, "
    "for three reasons: "
    "(1) block-design confound -- annotator groups label disjoint text "
    "blocks, so text-content features cluster annotators by group "
    "assignment, not behavior; "
    "(2) sparsity -- task2/3 labels exist only for YES labels, so "
    "low-yes-rate annotators give very noisy features; "
    "(3) circularity -- richer coupling between cluster definitions and the "
    "labels agents are evaluated on, while the task stays binary."
)


def normalized_size_entropy(sizes):
    """Shannon entropy of the cluster-size distribution / log2(k)."""
    p = np.array(sizes, dtype=float)
    p = p / p.sum()
    h = -(p * np.log2(p)).sum()
    return float(h / np.log2(len(p))) if len(p) > 1 else 0.0


def report_at_k(X_scaled, k, yes_rate, main_clusters):
    """Fit KMeans at k on the 2-feature space and summarize."""
    labels = fit_kmeans(X_scaled, k)
    names = name_clusters_by_yes_rate(labels, yes_rate)
    sil = float(silhouette_score(X_scaled, labels))

    keys = sorted(set(names))
    sizes = {c: names.count(c) for c in keys}
    n = len(names)
    yes_means = {c: float(np.mean([yes_rate[i] for i in range(n) if names[i] == c]))
                 for c in keys}
    means = sorted(yes_means.values())
    gaps = [means[i + 1] - means[i] for i in range(len(means) - 1)]
    min_gap = float(min(gaps)) if gaps else 0.0

    ari = float(adjusted_rand_score(main_clusters, names))
    crosstab = (
        pd.crosstab(pd.Series(names, name="ablation_2feat"),
                    pd.Series(list(main_clusters), name="main_3feat"))
        .to_dict()
    )
    # crosstab dict: {main_cluster: {ablation_cluster: count}}
    crosstab = {mc: {ac: int(v) for ac, v in col.items()}
                for mc, col in crosstab.items()}

    return {
        "k": int(k),
        "silhouette": round(sil, 4),
        "sizes": sizes,
        "min_share": round(min(sizes.values()) / n, 4),
        "normalized_size_entropy": round(normalized_size_entropy(list(sizes.values())), 4),
        "ari_vs_3feature": round(ari, 4),
        "crosstab_vs_3feature": crosstab,
        "cluster_mean_yes_rate": {c: round(v, 4) for c, v in yes_means.items()},
        "min_pairwise_yes_rate_gap": round(min_gap, 4),
        "all_pairwise_yes_rate_gaps": [round(g, 4) for g in gaps],
    }


def main(lang):
    print(f"=== Clustering ablation (2 features, no yes_rate) [{lang}] ===")

    # Main (3-feature) clustering, produced by behavioral_clustering.py
    main_csv = config.DATA_DIR / f"behavioral_features_{lang}.csv"
    if not main_csv.exists():
        raise FileNotFoundError(
            f"{main_csv} missing -- run behavioral_clustering.py --lang {lang} first")
    main_df = pd.read_csv(main_csv)
    with open(config.ARTIFACTS_DIR / "clustering" / lang / "cluster_summary.json") as f:
        main_summary = json.load(f)
    main_min_share = min(main_summary["sizes"].values()) / main_summary["n_annotators"]
    main_sil = main_summary["silhouette"]

    # Rebuild the label matrix / features so annotator order matches the CSV
    A, annotators, _ = build_label_matrix(lang)
    assert annotators == main_df["annotator_id"].tolist(), \
        "annotator order mismatch vs behavioral_features CSV"
    features, _counts = compute_behavioral_features(A)
    yes_rate = features[:, 0]
    X2 = features[:, 1:3]  # agreement_rate, label_entropy
    X2_scaled = StandardScaler().fit_transform(X2)

    main_clusters = main_df["cluster"].tolist()

    scan, best_k = scan_kmeans(X2_scaled, K_RANGE)
    print(f"2-feature best k = {best_k}")

    at_best = report_at_k(X2_scaled, best_k, yes_rate, main_clusters)
    at_3 = at_best if best_k == 3 else report_at_k(X2_scaled, 3, yes_rate, main_clusters)

    # ── Decision rule (evaluated at k=3) ──
    checks = {
        "min_share_ge_0.15": at_3["min_share"] >= 0.15,
        "min_share_gt_3feature": at_3["min_share"] > main_min_share,
        "silhouette_within_0.05_of_3feature": at_3["silhouette"] >= main_sil - 0.05,
        "all_yes_rate_gaps_ge_0.10": all(g >= 0.10 for g in at_3["all_pairwise_yes_rate_gaps"]),
    }
    recommendation = "adopt" if all(checks.values()) else "report-only"
    print(f"Recommendation: {recommendation} ({checks})")

    out = {
        "language": lang,
        "ablation": "kmeans on (agreement_rate, label_entropy) -- yes_rate dropped",
        "k_scan": scan,
        "best_k": int(best_k),
        "at_best_k": at_best,
        "at_k3": at_3,
        "reference_3feature": {
            "chosen_k": main_summary["chosen_k"],
            "silhouette": main_sil,
            "sizes": main_summary["sizes"],
            "min_share": round(main_min_share, 4),
        },
        "decision_checks_at_k3": checks,
        "recommendation": recommendation,
        "keyword_features": KEYWORD_FEATURES_NOTE,
    }

    out_path = config.RESULTS_DIR / f"clustering_ablation_{lang}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    main(ap.parse_args().lang)
