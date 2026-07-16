#!/usr/bin/env python3
"""
Preference Data Preparation Pipeline (language-parameterized)
=============================================================
Reads EXIST2024_training.json + data/behavioral_features_{lang}.csv (with
neutral cluster keys "cluster1".."clusterK"), and produces training data
for the per-cluster BehAv-PO agents.

Outputs (per language):
  data/splits/{lang}/split_info.json     Train/val/test group assignments
  data/splits/{lang}/train_set.json      Train texts + per-cluster votes
  data/splits/{lang}/test_set.json       Held-out texts + per-cluster votes
  data/splits/{lang}/cluster_meta.json   Cluster sizes/features + expected yes-rates
  data/preference/{lang}/sft_cluster{i}_{train,val}.jsonl
  data/preference/{lang}/dpo_cluster{i}_{train,val}.jsonl
  data/preference/{lang}/dataset_stats.json
  (--balance-labels) data/preference/{lang}/sft_balanced_cluster{i}_{train,val}.jsonl

The EN logic is a faithful re-parameterization of the frozen legacy run
(seed 42, same group-split ordering, same SFT mixing ratios, same DPO
pair construction): the new data/splits/en/{train,test}_set.json must
contain exactly the same text_ids as the legacy flat data/splits/*.json
(enforced by a hard guard). Legacy flat outputs are never rewritten.

Usage:
  python3 src/preprocessing/prepare_preference_data.py --lang en|es [--balance-labels]
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_exist_data(lang):
    """Load EXIST 2024 JSON and return texts for one language (file order)."""
    with open(config.DATA_PATH) as f:
        data = json.load(f)
    tag = config.LANG_SPLIT_TAG[lang]
    return [t for t in data.values() if tag in t.get("split", "")]


def load_cluster_assignments(lang):
    """Load per-language behavioral features CSV with cluster column."""
    path = config.DATA_DIR / f"behavioral_features_{lang}.csv"
    df = pd.read_csv(path)
    if "cluster" not in df.columns:
        raise ValueError(
            f"{path} missing 'cluster' column. "
            "Run behavioral_clustering.py first."
        )
    return dict(zip(df["annotator_id"], df["cluster"])), df


def build_text_records(texts, ann_to_cluster):
    """
    For each text, compute per-cluster majority votes and overall majority.

    Returns list of dicts with keys:
        text_id, tweet, group_key,
        cluster_votes  = {cluster: {"yes": n, "no": n, "majority": "YES"/"NO"/None}},
        overall_majority = "YES"/"NO"/None,
        clusters_present = set of cluster keys present
    """
    records = []
    for t in texts:
        text_id = t["id_EXIST"]
        tweet = t["tweet"]
        annotators = t["annotators"]
        labels = t["labels_task1"]
        group_key = tuple(sorted(annotators))

        # Per-cluster vote tallies
        cluster_votes = {}
        total_yes, total_no = 0, 0
        for ann_id, label_raw in zip(annotators, labels):
            cluster = ann_to_cluster.get(ann_id)
            if cluster is None:
                continue
            if cluster not in cluster_votes:
                cluster_votes[cluster] = {"yes": 0, "no": 0}
            if label_raw == "YES":
                cluster_votes[cluster]["yes"] += 1
                total_yes += 1
            elif label_raw == "NO":
                cluster_votes[cluster]["no"] += 1
                total_no += 1

        # Per-cluster majority
        for cluster, votes in cluster_votes.items():
            if votes["yes"] > votes["no"]:
                votes["majority"] = "YES"
            elif votes["no"] > votes["yes"]:
                votes["majority"] = "NO"
            else:
                votes["majority"] = None  # tie within cluster

            # Confidence: fraction of cluster annotators who agree with majority
            total = votes["yes"] + votes["no"]
            votes["confidence"] = max(votes["yes"], votes["no"]) / total if total > 0 else 0.5

        # Overall majority
        if total_yes > total_no:
            overall_majority = "YES"
        elif total_no > total_yes:
            overall_majority = "NO"
        else:
            overall_majority = None  # tie

        records.append({
            "text_id": text_id,
            "tweet": tweet,
            "group_key": group_key,
            "cluster_votes": cluster_votes,
            "overall_majority": overall_majority,
            "clusters_present": set(cluster_votes.keys()),
        })

    return records


def split_by_groups(records, lang):
    """
    Split records into train/val/test by annotator groups.
    Ensures no text leakage across splits. Group counts come from
    config.N_GROUP_SPLIT[lang]; ordering/seed identical to legacy.
    """
    # Collect unique groups
    groups = {}
    for r in records:
        gk = r["group_key"]
        if gk not in groups:
            groups[gk] = []
        groups[gk].append(r)

    group_keys = sorted(groups.keys(), key=lambda g: len(groups[g]), reverse=True)
    random.seed(config.RANDOM_STATE)
    random.shuffle(group_keys)

    n_train, n_val, n_test = config.N_GROUP_SPLIT[lang]
    if len(group_keys) != n_train + n_val + n_test:
        raise RuntimeError(
            f"[{lang}] found {len(group_keys)} annotator groups, "
            f"expected {n_train + n_val + n_test} (config.N_GROUP_SPLIT)")

    train_groups = group_keys[:n_train]
    val_groups = group_keys[n_train:n_train + n_val]
    test_groups = group_keys[n_train + n_val:]

    train = [r for gk in train_groups for r in groups[gk]]
    val = [r for gk in val_groups for r in groups[gk]]
    test = [r for gk in test_groups for r in groups[gk]]

    return train, val, test, train_groups, val_groups, test_groups


def make_chat_message(tweet, label):
    """Create an OpenAI chat fine-tuning example."""
    return {
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": f"Tweet: {tweet}"},
            {"role": "assistant", "content": label},
        ]
    }


def make_dpo_pair(tweet, preferred, non_preferred):
    """Create an OpenAI DPO preference pair (input.messages envelope)."""
    return {
        "input": {
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": f"Tweet: {tweet}"},
            ]
        },
        "preferred_output": [{"role": "assistant", "content": preferred}],
        "non_preferred_output": [{"role": "assistant", "content": non_preferred}],
    }


def _label_of(example):
    return example["messages"][2]["content"]


def balance_primary(primary):
    """Oversample the minority label among SFT primary examples.

    Uniform random draws (seed config.RANDOM_STATE) from the minority pool
    until minority share >= config.SFT_BALANCE_TARGET, with a per-example
    cap of config.SFT_BALANCE_MAX_DUP total copies. Returns (extras,
    achieved_minority_share).
    """
    n_yes = sum(1 for e in primary if _label_of(e) == "YES")
    n_no = len(primary) - n_yes
    if n_yes == 0 or n_no == 0 or len(primary) == 0:
        return [], 0.0
    minority = "YES" if n_yes < n_no else "NO"
    n_min = min(n_yes, n_no)
    pool = [e for e in primary if _label_of(e) == minority]

    rng = random.Random(config.RANDOM_STATE)
    copies = [1] * len(pool)
    extras = []
    while (n_min + len(extras)) / (len(primary) + len(extras)) < config.SFT_BALANCE_TARGET:
        eligible = [i for i in range(len(pool)) if copies[i] < config.SFT_BALANCE_MAX_DUP]
        if not eligible:
            break  # cap binds before target is reachable
        i = rng.choice(eligible)
        copies[i] += 1
        extras.append(pool[i])

    share = (n_min + len(extras)) / (len(primary) + len(extras))
    return extras, share


def build_sft_dataset(records, cluster_key, balance=False):
    """
    Build SFT dataset for a specific cluster agent.

    Includes:
    1. Primary: texts where this cluster has a clear majority vote
    2. Shared positive: texts where ALL present clusters agree (weighted)
    3. Team majority: texts with clear overall majority (weighted)

    With balance=True, the minority label is additionally oversampled among
    primary examples only (see balance_primary); the shared/team mixing is
    unchanged. Returns (dataset, balanced_share) where balanced_share is
    None when balance=False.
    """
    primary = []
    shared_positive = []
    team_majority = []

    for r in records:
        cv = r["cluster_votes"]
        tweet = r["tweet"]

        # 1. Primary: cluster's own majority label
        if cluster_key in cv and cv[cluster_key]["majority"] is not None:
            primary.append(make_chat_message(tweet, cv[cluster_key]["majority"]))

        # 2. Shared positive: all clusters present agree
        if len(r["clusters_present"]) >= 2:
            majorities = [
                cv[c]["majority"] for c in r["clusters_present"]
                if cv[c]["majority"] is not None
            ]
            if len(majorities) >= 2 and len(set(majorities)) == 1:
                shared_positive.append(make_chat_message(tweet, majorities[0]))

        # 3. Team majority
        if r["overall_majority"] is not None:
            team_majority.append(make_chat_message(tweet, r["overall_majority"]))

    # Mix in shared positive and team majority examples
    n_primary = len(primary)
    n_shared = int(n_primary * config.SFT_SHARED_POSITIVE_RATIO)
    n_team = int(n_primary * config.SFT_TEAM_MAJORITY_RATIO)

    random.seed(config.RANDOM_STATE)
    shared_sample = random.sample(shared_positive, min(n_shared, len(shared_positive)))
    team_sample = random.sample(team_majority, min(n_team, len(team_majority)))

    balanced_share = None
    extras = []
    if balance:
        # Separate RNG so the global stream (sampling + shuffle) is unchanged.
        extras, balanced_share = balance_primary(primary)

    dataset = primary + extras + shared_sample + team_sample
    random.shuffle(dataset)
    return dataset, balanced_share


def build_dpo_dataset(records, cluster_key):
    """
    Build DPO preference pairs for a specific cluster agent.

    For texts where this cluster's majority vote differs from at least
    one other cluster's vote:
        - preferred = this cluster's majority
        - non_preferred = opposite
    """
    pairs = []
    for r in records:
        cv = r["cluster_votes"]
        tweet = r["tweet"]

        if cluster_key not in cv:
            continue
        my_vote = cv[cluster_key]["majority"]
        if my_vote is None:
            continue

        # Check if any other cluster disagrees
        other_votes = [
            cv[c]["majority"] for c in r["clusters_present"]
            if c != cluster_key and cv[c]["majority"] is not None
        ]
        has_disagreement = any(v != my_vote for v in other_votes)

        if has_disagreement:
            opposite = "NO" if my_vote == "YES" else "YES"
            pairs.append(make_dpo_pair(tweet, my_vote, opposite))

    return pairs


def write_jsonl(data, path):
    """Write list of dicts to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def serialize_records(recs):
    return [{
        "text_id": r["text_id"],
        "tweet": r["tweet"],
        "cluster_votes": {
            c: {"majority": v["majority"], "yes": v["yes"], "no": v["no"]}
            for c, v in r["cluster_votes"].items()
        },
        "overall_majority": r["overall_majority"],
        "clusters_present": list(r["clusters_present"]),
    } for r in recs]


def build_cluster_meta(lang, features_df, cluster_keys, train):
    """Cluster sizes + behavioral feature means, and per-cluster expected
    yes-rates computed over TRAIN texts only (gold = cluster majority)."""
    clusters = {}
    for ck in cluster_keys:
        sub = features_df[features_df["cluster"] == ck]
        clusters[ck] = {
            "size": int(len(sub)),
            "yes_rate": round(float(sub["yes_rate"].mean()), 4),
            "agreement_rate": round(float(sub["agreement_rate"].mean()), 4),
            "label_entropy": round(float(sub["label_entropy"].mean()), 4),
        }

    expected_yes_rates = {}
    for ck in cluster_keys:
        majorities = [
            r["cluster_votes"][ck]["majority"] for r in train
            if ck in r["cluster_votes"]
            and r["cluster_votes"][ck]["majority"] is not None
        ]
        expected_yes_rates[ck] = (
            round(sum(1 for m in majorities if m == "YES") / len(majorities), 4)
            if majorities else None
        )

    return {
        "language": lang,
        "n_annotators": int(len(features_df)),
        "clusters": clusters,
        "expected_yes_rates": expected_yes_rates,
    }


def run_en_split_guard(splits_dir):
    """HARD GUARD: new EN splits must reproduce the frozen legacy splits."""
    for name, legacy_path in (("test_set", config.SPLITS_DIR / "test_set.json"),
                              ("train_set", config.SPLITS_DIR / "train_set.json")):
        with open(splits_dir / f"{name}.json") as f:
            new_ids = [r["text_id"] for r in json.load(f)]
        with open(legacy_path) as f:
            legacy_ids = [r["text_id"] for r in json.load(f)]
        if sorted(new_ids) != sorted(legacy_ids):
            print("\n" + "!" * 70)
            print(f"HARD GUARD FAILED (en): {name} text_ids differ from legacy")
            print(f"  new: {len(new_ids)} ids, legacy: {len(legacy_ids)} ids")
            print(f"  only-new: {sorted(set(new_ids) - set(legacy_ids))[:10]}")
            print(f"  only-legacy: {sorted(set(legacy_ids) - set(new_ids))[:10]}")
            print("  Debug ordering/seed until identical. DO NOT weaken this guard.")
            print("!" * 70)
            sys.exit(1)
        order_note = "identical order" if new_ids == legacy_ids else "same ids, different order"
        print(f"   EN split guard PASSED for {name} ({len(new_ids)} ids, {order_note})")


def main(lang, balance_labels=False):
    print("=" * 60)
    print(f"BehAv-PO Data Preparation Pipeline [{lang}]")
    print("=" * 60)

    splits_dir = config.SPLITS_DIR / lang
    pref_dir = config.PREFERENCE_DIR / lang
    splits_dir.mkdir(parents=True, exist_ok=True)
    pref_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"\n1. Loading EXIST 2024 data ({lang})...")
    texts = load_exist_data(lang)
    print(f"   Texts: {len(texts)}")

    print("\n2. Loading cluster assignments...")
    ann_to_cluster, features_df = load_cluster_assignments(lang)
    cluster_keys = sorted(set(ann_to_cluster.values()))
    cluster_counts = pd.Series(list(ann_to_cluster.values())).value_counts()
    print(f"   Annotators: {len(ann_to_cluster)}")
    for c in cluster_keys:
        print(f"   {c}: {cluster_counts[c]}")

    # Build text records
    print("\n3. Building text records with per-cluster votes...")
    records = build_text_records(texts, ann_to_cluster)
    print(f"   Total records: {len(records)}")

    # Cross-cluster coverage
    multi_cluster = sum(1 for r in records if len(r["clusters_present"]) >= 2)
    print(f"   Texts with >=2 clusters: {multi_cluster}")

    # Disagreement counts
    disagree = 0
    for r in records:
        majorities = [
            r["cluster_votes"][c]["majority"]
            for c in r["clusters_present"]
            if r["cluster_votes"][c]["majority"] is not None
        ]
        if len(set(majorities)) > 1:
            disagree += 1
    print(f"   Texts with cross-cluster disagreement: {disagree}")

    # Split
    print("\n4. Splitting by annotator groups...")
    train, val, test, train_groups, val_groups, test_groups = split_by_groups(records, lang)
    print(f"   Train: {len(train)} texts ({len(train_groups)} groups)")
    print(f"   Val:   {len(val)} texts ({len(val_groups)} groups)")
    print(f"   Test:  {len(test)} texts ({len(test_groups)} groups)")

    # Build datasets per cluster
    print("\n5. Building SFT and DPO datasets...")
    stats = {"language": lang,
             "splits": {"train": len(train), "val": len(val), "test": len(test)}}

    for ck in cluster_keys:
        # SFT
        sft_train, _ = build_sft_dataset(train, ck)
        sft_val, _ = build_sft_dataset(val, ck)
        write_jsonl(sft_train, str(pref_dir / f"sft_{ck}_train.jsonl"))
        write_jsonl(sft_val, str(pref_dir / f"sft_{ck}_val.jsonl"))

        # DPO
        dpo_train = build_dpo_dataset(train, ck)
        dpo_val = build_dpo_dataset(val, ck)
        write_jsonl(dpo_train, str(pref_dir / f"dpo_{ck}_train.jsonl"))
        write_jsonl(dpo_val, str(pref_dir / f"dpo_{ck}_val.jsonl"))

        print(f"\n   {ck}:")
        print(f"     SFT train: {len(sft_train)}, val: {len(sft_val)}")
        print(f"     DPO train: {len(dpo_train)}, val: {len(dpo_val)}")

        stats[ck] = {
            "sft_train": len(sft_train),
            "sft_val": len(sft_val),
            "dpo_train": len(dpo_train),
            "dpo_val": len(dpo_val),
        }

        # Balanced SFT ablation (oversample minority label in primary only)
        if balance_labels:
            bal_train, share_train = build_sft_dataset(train, ck, balance=True)
            bal_val, share_val = build_sft_dataset(val, ck, balance=True)
            write_jsonl(bal_train, str(pref_dir / f"sft_balanced_{ck}_train.jsonl"))
            write_jsonl(bal_val, str(pref_dir / f"sft_balanced_{ck}_val.jsonl"))
            print(f"     SFT balanced train: {len(bal_train)} "
                  f"(primary minority share {share_train:.3f}), "
                  f"val: {len(bal_val)} (share {share_val:.3f})")
            stats[ck].update({
                "sft_balanced_train": len(bal_train),
                "sft_balanced_val": len(bal_val),
                "balanced_minority_share_train": round(share_train, 4),
                "balanced_minority_share_val": round(share_val, 4),
            })

    # Save split info
    split_info = {
        "language": lang,
        "train_groups": [list(g) for g in train_groups],
        "val_groups": [list(g) for g in val_groups],
        "test_groups": [list(g) for g in test_groups],
        "n_train_texts": len(train),
        "n_val_texts": len(val),
        "n_test_texts": len(test),
    }
    with open(splits_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Save test set for evaluation, train set for GRPO reward computation
    with open(splits_dir / "test_set.json", "w") as f:
        json.dump(serialize_records(test), f, indent=2, ensure_ascii=False)
    with open(splits_dir / "train_set.json", "w") as f:
        json.dump(serialize_records(train), f, indent=2, ensure_ascii=False)

    # Cluster metadata (sizes, behavioral means, expected yes-rates on TRAIN)
    cluster_meta = build_cluster_meta(lang, features_df, cluster_keys, train)
    with open(splits_dir / "cluster_meta.json", "w") as f:
        json.dump(cluster_meta, f, indent=2)
    print("\n   Expected yes-rates (train):", cluster_meta["expected_yes_rates"])

    # Save stats
    with open(pref_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    # Hard guard: EN must reproduce the frozen legacy split exactly
    if lang == "en":
        print("\n6. Verifying EN splits against legacy flat files...")
        run_en_split_guard(splits_dir)

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"Output directories: {splits_dir}/, {pref_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--balance-labels", action="store_true",
                    help="also write sft_balanced_* files (minority-label "
                         "oversampling in primary examples)")
    args = ap.parse_args()
    main(args.lang, balance_labels=args.balance_labels)
