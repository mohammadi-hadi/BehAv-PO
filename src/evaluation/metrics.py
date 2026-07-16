#!/usr/bin/env python3
"""
Evaluation Metrics
==================
Metrics for the BehAv-PO multi-agent system:
  - CAA  (Cluster-Alignment Accuracy): agent matches its cluster's vote
  - MVA  (Majority Vote Accuracy): team vote matches overall majority
  - YES-Rate Calibration: |agent_yes_rate - cluster_yes_rate|
  - Disagreement Rate: fraction of texts where agents disagree
  - F1-macro: team prediction F1
"""

import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def cluster_alignment_accuracy(predictions, ground_truth_cluster_votes, cluster_name):
    """
    CAA: fraction of texts where agent's prediction matches
    the cluster's majority vote.

    Args:
        predictions: list of "YES"/"NO" from the agent
        ground_truth_cluster_votes: list of dicts with cluster vote info per text
        cluster_name: which cluster this agent represents

    Returns:
        float: accuracy (0-1), or None if no applicable texts
    """
    correct, total = 0, 0
    for pred, gv in zip(predictions, ground_truth_cluster_votes):
        if cluster_name not in gv:
            continue
        cluster_majority = gv[cluster_name].get("majority")
        if cluster_majority is None:
            continue
        total += 1
        if pred == cluster_majority:
            correct += 1
    return correct / total if total > 0 else None


def majority_vote_accuracy(team_votes, overall_majorities):
    """
    MVA: fraction of texts where team majority vote matches
    the overall human majority label.

    Args:
        team_votes: list of "YES"/"NO" team predictions
        overall_majorities: list of "YES"/"NO"/None overall majority labels

    Returns:
        float: accuracy (0-1), or None if no applicable texts
    """
    correct, total = 0, 0
    for vote, majority in zip(team_votes, overall_majorities):
        if majority is None:
            continue
        total += 1
        if vote == majority:
            correct += 1
    return correct / total if total > 0 else None


def yes_rate_calibration(predictions, cluster_name):
    """
    Calibration: |agent_yes_rate - expected_cluster_yes_rate|

    Args:
        predictions: list of "YES"/"NO"
        cluster_name: cluster this agent represents

    Returns:
        dict with agent_yes_rate, expected_yes_rate, calibration_error
    """
    if not predictions:
        return None
    yes_count = sum(1 for p in predictions if p == "YES")
    agent_yes_rate = yes_count / len(predictions)
    expected = config.EXPECTED_YES_RATES.get(cluster_name, 0.5)
    return {
        "agent_yes_rate": round(agent_yes_rate, 4),
        "expected_yes_rate": expected,
        "calibration_error": round(abs(agent_yes_rate - expected), 4),
    }


def disagreement_rate(all_agent_predictions):
    """
    Fraction of texts where at least 2 agents disagree.

    Args:
        all_agent_predictions: list of dicts {cluster: "YES"/"NO"} per text

    Returns:
        float: disagreement rate (0-1)
    """
    if not all_agent_predictions:
        return 0.0
    disagree = 0
    for preds in all_agent_predictions:
        labels = list(preds.values())
        if len(set(labels)) > 1:
            disagree += 1
    return disagree / len(all_agent_predictions)


def f1_macro(team_votes, overall_majorities):
    """
    Macro-averaged F1 for team predictions vs overall majority.

    Args:
        team_votes: list of "YES"/"NO"
        overall_majorities: list of "YES"/"NO"/None

    Returns:
        dict with f1_yes, f1_no, f1_macro
    """
    # Filter out texts with no clear majority
    pairs = [(v, m) for v, m in zip(team_votes, overall_majorities) if m is not None]
    if not pairs:
        return None

    tp = {"YES": 0, "NO": 0}
    fp = {"YES": 0, "NO": 0}
    fn = {"YES": 0, "NO": 0}

    for pred, gold in pairs:
        for label in ["YES", "NO"]:
            if pred == label and gold == label:
                tp[label] += 1
            elif pred == label and gold != label:
                fp[label] += 1
            elif pred != label and gold == label:
                fn[label] += 1

    f1s = {}
    for label in ["YES", "NO"]:
        precision = tp[label] / (tp[label] + fp[label]) if (tp[label] + fp[label]) > 0 else 0
        recall = tp[label] / (tp[label] + fn[label]) if (tp[label] + fn[label]) > 0 else 0
        f1s[f"f1_{label.lower()}"] = round(
            2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0, 4
        )

    f1s["f1_macro"] = round(np.mean([f1s["f1_yes"], f1s["f1_no"]]), 4)
    return f1s


def compute_all_metrics(system_results, test_records):
    """
    Compute all metrics from system results and test records.

    Args:
        system_results: list of dicts from MultiAgentSystem.predict_batch()
        test_records: list of dicts from test_set.json

    Returns:
        dict with all metrics
    """
    metrics = {}

    # Extract predictions
    team_votes = [r["team_vote"] for r in system_results]
    overall_majorities = [t["overall_majority"] for t in test_records]
    all_agent_preds = [r["agent_predictions"] for r in system_results]
    cluster_votes_list = [t["cluster_votes"] for t in test_records]

    # MVA
    metrics["mva"] = majority_vote_accuracy(team_votes, overall_majorities)

    # F1
    f1 = f1_macro(team_votes, overall_majorities)
    if f1:
        metrics.update(f1)

    # Disagreement rate
    metrics["disagreement_rate"] = round(disagreement_rate(all_agent_preds), 4)

    # Per-cluster metrics
    for cluster in config.CLUSTER_NAMES:
        agent_preds = [r["agent_predictions"][cluster] for r in system_results]

        # CAA
        caa = cluster_alignment_accuracy(agent_preds, cluster_votes_list, cluster)
        metrics[f"caa_{cluster.lower()}"] = round(caa, 4) if caa is not None else None

        # Calibration
        cal = yes_rate_calibration(agent_preds, cluster)
        if cal:
            metrics[f"yes_rate_{cluster.lower()}"] = cal["agent_yes_rate"]
            metrics[f"calibration_{cluster.lower()}"] = cal["calibration_error"]

    return metrics
