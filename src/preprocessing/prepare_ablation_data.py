#!/usr/bin/env python3
"""
Ablation Dataset Preparation
=============================
Generates datasets for ablation experiments:

1. SFT-pure: Pure cluster labels, no shared positive or team majority mixing
2. DPO-team: Preference pairs on unanimous-agreement texts (team signal)
3. DPO-marspo: Combined individual (disagreement) + team (agreement) pairs

Usage:
    python data/prepare_ablation_data.py
"""

import json
import os
import random
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_exist_data():
    with open(config.DATA_PATH) as f:
        data = json.load(f)
    return [t for t in data.values() if "EN" in t.get("split", "")]


def load_cluster_assignments():
    df = pd.read_csv(config.BEHAVIORAL_FEATURES_PATH)
    return dict(zip(df["annotator_id"], df["cluster"]))


def build_text_records(en_texts, ann_to_cluster):
    """Same as prepare_preference_data.py but inline."""
    records = []
    for t in en_texts:
        cluster_votes = {}
        total_yes, total_no = 0, 0
        for ann_id, label_raw in zip(t["annotators"], t["labels_task1"]):
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

        for cluster, v in cluster_votes.items():
            if v["yes"] > v["no"]:
                v["majority"] = "YES"
            elif v["no"] > v["yes"]:
                v["majority"] = "NO"
            else:
                v["majority"] = None

        if total_yes > total_no:
            overall = "YES"
        elif total_no > total_yes:
            overall = "NO"
        else:
            overall = None

        records.append({
            "text_id": t["id_EXIST"],
            "tweet": t["tweet"],
            "group_key": tuple(sorted(t["annotators"])),
            "cluster_votes": cluster_votes,
            "overall_majority": overall,
            "clusters_present": set(cluster_votes.keys()),
        })
    return records


def load_split_info():
    with open(os.path.join(config.DATA_DIR, "split_info.json")) as f:
        info = json.load(f)
    train_groups = set(tuple(g) for g in info["train_groups"])
    val_groups = set(tuple(g) for g in info["val_groups"])
    test_groups = set(tuple(g) for g in info["test_groups"])
    return train_groups, val_groups, test_groups


def make_chat(tweet, label):
    return {
        "messages": [
            {"role": "system", "content": config.SYSTEM_PROMPT},
            {"role": "user", "content": f"Tweet: {tweet}"},
            {"role": "assistant", "content": label},
        ]
    }


def make_dpo(tweet, preferred, non_preferred):
    return {
        "input": {
            "messages": [
                {"role": "system", "content": config.SYSTEM_PROMPT},
                {"role": "user", "content": f"Tweet: {tweet}"},
            ],
        },
        "preferred_output": [{"role": "assistant", "content": preferred}],
        "non_preferred_output": [{"role": "assistant", "content": non_preferred}],
    }


def write_jsonl(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


# =============================================================
# 1. SFT-pure: only cluster's own labels, no mixing
# =============================================================
def build_sft_pure(records, cluster_name):
    data = []
    for r in records:
        cv = r["cluster_votes"]
        if cluster_name in cv and cv[cluster_name]["majority"] is not None:
            data.append(make_chat(r["tweet"], cv[cluster_name]["majority"]))
    random.shuffle(data)
    return data


# =============================================================
# 2. DPO-team: preference pairs on unanimous-agreement texts
# =============================================================
def build_dpo_team(records):
    """
    Preference pairs where the preferred label is the unanimous agreement
    across ALL clusters present (2+ clusters). Shared across all agents.
    """
    pairs = []
    for r in records:
        cv = r["cluster_votes"]
        if len(r["clusters_present"]) < 2:
            continue

        majorities = [
            cv[c]["majority"] for c in r["clusters_present"]
            if cv[c]["majority"] is not None
        ]
        if len(majorities) < 2 or len(set(majorities)) != 1:
            continue

        unanimous = majorities[0]
        opposite = "NO" if unanimous == "YES" else "YES"
        pairs.append(make_dpo(r["tweet"], unanimous, opposite))

    return pairs


# =============================================================
# 3. DPO-marspo: combined individual (disagreement) + team (agreement)
# =============================================================
def build_dpo_marspo(records, cluster_name, team_pairs):
    """
    For each cluster agent:
      - Individual pairs (on cluster-disagreement texts) = cluster's view
      - Team pairs (on unanimous-agreement texts) = team consensus
    Combined.
    """
    individual = []
    for r in records:
        cv = r["cluster_votes"]
        if cluster_name not in cv:
            continue
        my_vote = cv[cluster_name]["majority"]
        if my_vote is None:
            continue

        other_votes = [
            cv[c]["majority"] for c in r["clusters_present"]
            if c != cluster_name and cv[c]["majority"] is not None
        ]
        has_disagreement = any(v != my_vote for v in other_votes)

        if has_disagreement:
            opposite = "NO" if my_vote == "YES" else "YES"
            individual.append(make_dpo(r["tweet"], my_vote, opposite))

    combined = individual + team_pairs
    random.shuffle(combined)
    return combined, len(individual)


def main():
    print("=" * 60)
    print("Ablation Dataset Preparation")
    print("=" * 60)

    en_texts = load_exist_data()
    ann_to_cluster = load_cluster_assignments()
    records = build_text_records(en_texts, ann_to_cluster)
    print(f"\nTotal records: {len(records)}")

    train_groups, val_groups, test_groups = load_split_info()
    train = [r for r in records if r["group_key"] in train_groups]
    val = [r for r in records if r["group_key"] in val_groups]
    print(f"Train: {len(train)} texts")
    print(f"Val: {len(val)} texts")

    # Build shared team DPO pairs
    print("\n--- Shared team DPO pairs (unanimous agreement texts) ---")
    team_train = build_dpo_team(train)
    team_val = build_dpo_team(val)
    write_jsonl(team_train, "data/preference/dpo_team_train.jsonl")
    write_jsonl(team_val, "data/preference/dpo_team_val.jsonl")
    print(f"  Team train: {len(team_train)} pairs")
    print(f"  Team val: {len(team_val)} pairs")

    # Per-cluster datasets
    stats = {"team_train": len(team_train), "team_val": len(team_val)}
    for cluster in config.CLUSTER_NAMES:
        c_lower = cluster.lower()
        print(f"\n--- {cluster} ---")

        # SFT-pure
        sft_pure_train = build_sft_pure(train, cluster)
        sft_pure_val = build_sft_pure(val, cluster)
        write_jsonl(sft_pure_train, f"data/preference/sft_pure_{c_lower}_train.jsonl")
        write_jsonl(sft_pure_val, f"data/preference/sft_pure_{c_lower}_val.jsonl")
        print(f"  SFT-pure train: {len(sft_pure_train)}")
        print(f"  SFT-pure val: {len(sft_pure_val)}")

        # DPO-marspo (individual + team)
        marspo_train, n_indiv = build_dpo_marspo(train, cluster, team_train)
        marspo_val, n_indiv_val = build_dpo_marspo(val, cluster, team_val)
        write_jsonl(marspo_train, f"data/preference/dpo_marspo_{c_lower}_train.jsonl")
        write_jsonl(marspo_val, f"data/preference/dpo_marspo_{c_lower}_val.jsonl")
        print(f"  Mars-PO train: {len(marspo_train)} ({n_indiv} indiv + {len(team_train)} team)")
        print(f"  Mars-PO val: {len(marspo_val)} ({n_indiv_val} indiv + {len(team_val)} team)")

        stats[cluster] = {
            "sft_pure_train": len(sft_pure_train),
            "sft_pure_val": len(sft_pure_val),
            "marspo_train": len(marspo_train),
            "marspo_indiv_train": n_indiv,
            "marspo_val": len(marspo_val),
        }

    with open("data/preference/ablation_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 60)
    print("Ablation data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    random.seed(config.RANDOM_STATE)
    main()
