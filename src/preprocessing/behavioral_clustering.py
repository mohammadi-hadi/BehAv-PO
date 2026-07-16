#!/usr/bin/env python3
"""
Behavioral Annotator Clustering (language-parameterized)
=========================================================
Re-implements ONLY the behavioral-clustering core of the legacy
`annotator_clustering_pipeline.py` (Sections 1 + 2B), parameterized by
language, so the identical procedure can be applied to the Spanish
subset of EXIST 2024.

Procedure (must match the legacy pipeline exactly for lang=en):
  1. Build annotator x text label matrix (+1 YES / -1 NO / 0 missing)
     from texts with entry["lang"] == lang; keep annotators with
     >= MIN_LABELS labels.
  2. Per-annotator features: yes_rate, agreement_rate (vs per-text
     majority sign of column sums, excluding zero-majority texts),
     label_entropy (binary Shannon entropy, base 2).
  3. StandardScaler + KMeans(n_init=20, random_state=42), scan k=2..20,
     pick k by max silhouette (rounded to 4 decimals, first argmax --
     matches legacy idxmax on rounded values); refit at best k.
  4. Name clusters by ascending mean yes_rate: "cluster1".."clusterK".

Outputs:
  data/behavioral_features_{lang}.csv
  artifacts/clustering/{lang}/kmeans_scan.json
  artifacts/clustering/{lang}/cluster_summary.json

Hard guard (lang=en): assignments must match the frozen legacy
data/behavioral_features.csv exactly (ARI == 1.0, sizes 75/224/49).

Usage:
  python3 src/preprocessing/behavioral_clustering.py --lang en|es
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

FEATURE_NAMES = ["yes_rate", "agreement_rate", "label_entropy"]
K_RANGE = range(2, 21)


def build_label_matrix(lang):
    """Annotator x text matrix A (+1 YES / -1 NO / 0 missing) for one language.

    Matches the legacy pipeline: pivot to text x annotator, fillna(0),
    keep annotators with >= MIN_LABELS non-zero entries (column order =
    sorted annotator_id, as produced by pivot_table).

    Returns (A, annotator_names, text_ids).
    """
    with open(config.DATA_PATH) as f:
        raw = json.load(f)

    rows = []
    for entry in raw.values():
        if entry["lang"] != lang:
            continue
        labels = entry["labels_task1"]
        for i, ann in enumerate(entry["annotators"]):
            lab = labels[i] if i < len(labels) else None
            rows.append({
                "text_id": entry["id_EXIST"],
                "annotator_id": ann,
                "label": 1 if lab == "YES" else (-1 if lab == "NO" else 0),
            })

    annot_df = pd.DataFrame(rows)
    matrix = (
        annot_df
        .pivot_table(index="text_id", columns="annotator_id",
                     values="label", aggfunc="first")
        .fillna(0)
        .astype(int)
    )

    non_zero_counts = (matrix != 0).sum(axis=0)
    active = non_zero_counts[non_zero_counts >= config.MIN_LABELS].index.tolist()
    A = matrix[active].values.T  # (n_annotators, n_texts)
    return A, active, matrix.index.tolist()


def compute_behavioral_features(A):
    """Per-annotator (yes_rate, agreement_rate, label_entropy) + counts.

    Identical math to legacy Section 2B (including agreement default 0.5
    for annotators with no majority-decided texts).
    """
    n_annotators, n_texts = A.shape
    n_yes = (A == 1).sum(axis=1)
    n_no = (A == -1).sum(axis=1)
    n_labeled = (A != 0).sum(axis=1)
    yes_rate = n_yes / n_labeled

    # Agreement with per-text majority (sign of column sums over active
    # annotators); texts with zero majority (tie) are excluded.
    majority = np.sign(A.sum(axis=0))
    valid = (A != 0) & (majority != 0)[None, :]
    agree = ((A == majority[None, :]) & valid).sum(axis=1).astype(float)
    total = valid.sum(axis=1).astype(float)
    agreement_rate = np.where(total > 0, agree / np.maximum(total, 1.0), 0.5)

    # Binary Shannon entropy (base 2) of YES/NO proportions.
    p_yes = n_yes / n_labeled
    p_no = n_no / n_labeled
    entropy = np.zeros(n_annotators)
    for p in (p_yes, p_no):  # same term order as legacy loop
        mask = p > 0
        entropy[mask] -= p[mask] * np.log2(p[mask])

    features = np.column_stack([yes_rate, agreement_rate, entropy])
    counts = {"n_labeled": n_labeled, "n_yes": n_yes, "n_no": n_no}
    return features, counts


def scan_kmeans(X_scaled, k_range=K_RANGE):
    """Silhouette scan; returns (scan_rows, best_k). Selection mimics the
    legacy pipeline: round silhouette to 4 decimals, take first argmax."""
    scan = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=20)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        scan.append({"k": k, "silhouette": round(float(sil), 4),
                     "min_cluster_size": int(np.bincount(labels).min())})
    sils = np.array([r["silhouette"] for r in scan])
    best_k = scan[int(np.argmax(sils))]["k"]
    return scan, best_k


def fit_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, random_state=config.RANDOM_STATE, n_init=20)
    return km.fit_predict(X_scaled)


def name_clusters_by_yes_rate(labels, yes_rate):
    """Map raw KMeans labels -> 'cluster1'..'clusterK' by ascending mean
    yes_rate. Returns list of string names aligned with labels."""
    ks = np.unique(labels)
    means = {c: yes_rate[labels == c].mean() for c in ks}
    order = sorted(ks, key=lambda c: means[c])
    name_map = {c: f"cluster{i + 1}" for i, c in enumerate(order)}
    return [name_map[c] for c in labels]


def cluster_behavioral(lang, k_range=K_RANGE, force_k=None):
    """Full clustering for one language. Returns a dict of results.

    force_k: fit at this k regardless of the silhouette scan (used for
    architecture parity, e.g. ES forced to 3 clusters via config.FORCE_K).
    The scan facts (silhouette-best k) are still recorded.
    """
    A, annotators, text_ids = build_label_matrix(lang)
    expected = config.EXPECTED_ANNOTATORS.get(lang)
    if expected is not None and len(annotators) != expected:
        raise RuntimeError(
            f"[{lang}] expected {expected} active annotators, got {len(annotators)}")

    features, counts = compute_behavioral_features(A)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)

    scan, best_k = scan_kmeans(X_scaled, k_range)
    chosen_k = force_k if force_k is not None else best_k
    labels = fit_kmeans(X_scaled, chosen_k)
    names = name_clusters_by_yes_rate(labels, features[:, 0])
    sil = float(silhouette_score(X_scaled, labels))

    df = pd.DataFrame({
        "annotator_id": annotators,
        "yes_rate": features[:, 0],
        "agreement_rate": features[:, 1],
        "label_entropy": features[:, 2],
        "n_labeled": counts["n_labeled"],
        "n_yes": counts["n_yes"],
        "n_no": counts["n_no"],
        "cluster": names,
    })
    best_sil = next(r["silhouette"] for r in scan if r["k"] == best_k)
    return {"df": df, "scan": scan, "best_k": best_k, "best_silhouette": best_sil,
            "chosen_k": chosen_k, "forced": force_k is not None,
            "silhouette": sil, "n_texts": A.shape[1]}


def run_en_guard(new_df):
    """HARD GUARD: new EN assignment must reproduce the frozen legacy one."""
    legacy = pd.read_csv(config.BEHAVIORAL_FEATURES_PATH)
    legacy["cluster_key"] = legacy["cluster"].map(config.LEGACY_CLUSTER_TO_KEY)
    merged = legacy[["annotator_id", "cluster_key"]].merge(
        new_df[["annotator_id", "cluster"]], on="annotator_id", how="outer")
    if merged.isna().any().any() or len(merged) != len(legacy):
        print("\n" + "!" * 70)
        print("HARD GUARD FAILED (en): annotator sets differ from legacy CSV")
        print("!" * 70)
        sys.exit(1)
    ari = adjusted_rand_score(merged["cluster_key"], merged["cluster"])
    sizes = new_df["cluster"].value_counts().to_dict()
    expected_sizes = {"cluster1": 75, "cluster2": 224, "cluster3": 49}
    if not np.isclose(ari, 1.0) or sizes != expected_sizes:
        print("\n" + "!" * 70)
        print("HARD GUARD FAILED (en): new clustering does not match legacy")
        print(f"  ARI vs legacy = {ari:.6f} (must be 1.0)")
        print(f"  sizes = {sizes} (must be {expected_sizes})")
        print("  Debug the feature/matrix computation until it matches the")
        print("  legacy pipeline. DO NOT weaken this guard.")
        print("!" * 70)
        sys.exit(1)
    print(f"EN guard PASSED: ARI = {ari:.4f}, sizes = {expected_sizes}")
    return ari


def main(lang, force_k=None):
    print(f"=== Behavioral clustering [{lang}] ===")
    if force_k is None:
        force_k = config.FORCE_K.get(lang)
    res = cluster_behavioral(lang, force_k=force_k)
    df, scan, best_k, sil = res["df"], res["scan"], res["best_k"], res["silhouette"]
    chosen_k = res["chosen_k"]
    print(f"Texts: {res['n_texts']}, annotators: {len(df)}")
    if res["forced"]:
        print(f"Silhouette-best k = {best_k} (silhouette = {res['best_silhouette']:.4f})")
        print(f"FORCED k = {chosen_k} (silhouette at k={chosen_k}: {sil:.4f})")
    else:
        print(f"Best k = {best_k} (silhouette = {sil:.4f})")

    sizes = df["cluster"].value_counts().sort_index()
    for cname, n in sizes.items():
        m = df[df["cluster"] == cname]
        print(f"  {cname}: n={n}, yes_rate={m['yes_rate'].mean():.3f}, "
              f"agreement={m['agreement_rate'].mean():.3f}, "
              f"entropy={m['label_entropy'].mean():.3f}")

    if lang == "en":
        run_en_guard(df)

    # ── Outputs ──
    csv_path = config.DATA_DIR / f"behavioral_features_{lang}.csv"
    df.to_csv(csv_path, index=False)

    out_dir = config.ARTIFACTS_DIR / "clustering" / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "kmeans_scan.json", "w") as f:
        json.dump({"language": lang, "k_range": [min(K_RANGE), max(K_RANGE)],
                   "scan": scan, "best_k": best_k}, f, indent=2)

    summary = {
        "language": lang,
        "chosen_k": chosen_k,
        "forced": res["forced"],
        "silhouette_best_k": best_k,
        "silhouette_best": res["best_silhouette"],
        "silhouette_at_chosen": round(sil, 4),
        "silhouette": round(sil, 4),
        "n_annotators": len(df),
        "n_texts": res["n_texts"],
        "sizes": {c: int(n) for c, n in sizes.items()},
        "features": {
            c: {feat: {"mean": round(float(df.loc[df["cluster"] == c, feat].mean()), 4),
                       "std": round(float(df.loc[df["cluster"] == c, feat].std()), 4)}
                for feat in FEATURE_NAMES}
            for c in sizes.index
        },
    }
    with open(out_dir / "cluster_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir}/kmeans_scan.json, cluster_summary.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--force-k", type=int, default=None,
                    help="fit at this k regardless of the silhouette scan "
                         "(default: config.FORCE_K[lang])")
    args = ap.parse_args()
    main(args.lang, force_k=args.force_k)
