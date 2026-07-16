#!/usr/bin/env python3
"""
Evaluate the local Qwen3-8B multi-agent system on a per-language test set.

Always saves per-text predictions first (including p(YES) per agent), then
computes every metric from the saved predictions, so any metric can be
recomputed later without re-running inference.

Result schema (all cluster keys neutral):
  core: mva, f1_yes, f1_no, f1_macro, balanced_accuracy, disagreement_rate,
        caa_cluster{i}, yes_rate_cluster{i}, calibration_cluster{i},
        f1_macro_cluster{i}, balanced_accuracy_cluster{i}
  prevalence: per-cluster n_present / n_majority / n_tied / yes_rate
  all_clusters_subset + two_plus_clusters_subset: same core metrics on the
        test texts where all 3 (>= 2) clusters have annotators, with Wilson
        95% CIs on mva and balanced accuracy (small n).

Run under .venv-mlx.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

STAGES = ["base", "sft", "dpo", "grpo_a20", "sft_balanced"]


def adapter_paths(lang, stage):
    if stage == "base":
        return {ck: None for ck in config.CLUSTER_KEYS}
    return {ck: config.ADAPTERS_DIR / lang / f"{stage}_{ck}"
            for ck in config.CLUSTER_KEYS}


def wilson_ci(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


def binary_metrics(golds, preds):
    """accuracy, f1_yes, f1_no, f1_macro, balanced_accuracy over aligned
    YES/NO lists (pairs with gold None are skipped by the caller)."""
    tp = sum(1 for g, p in zip(golds, preds) if g == "YES" and p == "YES")
    fp = sum(1 for g, p in zip(golds, preds) if g == "NO" and p == "YES")
    fn = sum(1 for g, p in zip(golds, preds) if g == "YES" and p == "NO")
    tn = sum(1 for g, p in zip(golds, preds) if g == "NO" and p == "NO")
    n = tp + fp + fn + tn
    if n == 0:
        return None
    f1_yes, f1_no = _f1(tp, fp, fn), _f1(tn, fn, fp)
    rec_yes = tp / (tp + fn) if (tp + fn) else 0.0
    rec_no = tn / (tn + fp) if (tn + fp) else 0.0
    return {
        "n": n,
        "accuracy": round((tp + tn) / n, 4),
        "f1_yes": round(f1_yes, 4),
        "f1_no": round(f1_no, 4),
        "f1_macro": round((f1_yes + f1_no) / 2, 4),
        "balanced_accuracy": round((rec_yes + rec_no) / 2, 4),
    }


def core_metrics(preds, records, expected_yes_rates):
    """Full metric block over aligned (predictions, test records)."""
    out = {}
    team_golds = [r.get("overall_majority") for r in records]
    team_votes = [p["team_vote"] for p in preds]
    pairs = [(g, v) for g, v in zip(team_golds, team_votes) if g is not None]
    team = binary_metrics([g for g, _ in pairs], [v for _, v in pairs])
    if team:
        out.update({"n_texts": len(records), "n_gold": team["n"],
                    "mva": team["accuracy"], "f1_yes": team["f1_yes"],
                    "f1_no": team["f1_no"], "f1_macro": team["f1_macro"],
                    "balanced_accuracy": team["balanced_accuracy"]})
    out["disagreement_rate"] = round(
        sum(1 for p in preds if not p["unanimous"]) / len(preds), 4) if preds else None

    prevalence = {"overall": {
        "n": len(records),
        "n_gold": sum(1 for g in team_golds if g is not None),
        "yes_rate": round(sum(1 for g in team_golds if g == "YES")
                          / max(1, sum(1 for g in team_golds if g is not None)), 4),
    }}
    for ck in config.CLUSTER_KEYS:
        golds, agent_preds = [], []
        n_present = n_majority = n_tied = n_yes_gold = 0
        for p, r in zip(preds, records):
            cv = r.get("cluster_votes", {}).get(ck)
            ap = p["agent_predictions"].get(ck)
            if cv is not None:
                n_present += 1
                if cv.get("majority") is None:
                    n_tied += 1
                else:
                    n_majority += 1
                    n_yes_gold += cv["majority"] == "YES"
                    if ap is not None:
                        golds.append(cv["majority"])
                        agent_preds.append(ap)
        m = binary_metrics(golds, agent_preds)
        yes_rate = round(sum(1 for p in preds
                             if p["agent_predictions"].get(ck) == "YES")
                         / len(preds), 4) if preds else None
        out[f"caa_{ck}"] = m["accuracy"] if m else None
        out[f"f1_macro_{ck}"] = m["f1_macro"] if m else None
        out[f"balanced_accuracy_{ck}"] = m["balanced_accuracy"] if m else None
        out[f"yes_rate_{ck}"] = yes_rate
        target = expected_yes_rates.get(ck)
        out[f"calibration_{ck}"] = (round(abs(yes_rate - target), 4)
                                    if yes_rate is not None and target is not None
                                    else None)
        prevalence[ck] = {"n_present": n_present, "n_majority": n_majority,
                          "n_tied": n_tied,
                          "yes_rate": round(n_yes_gold / n_majority, 4)
                          if n_majority else None}
    out["prevalence"] = prevalence
    return out


def subset_metrics(preds, records, expected_yes_rates, min_clusters):
    keep = [i for i, r in enumerate(records)
            if len(r.get("cluster_votes", {})) >= min_clusters]
    sub_preds = [preds[i] for i in keep]
    sub_recs = [records[i] for i in keep]
    if not sub_recs:
        return None
    m = core_metrics(sub_preds, sub_recs, expected_yes_rates)
    if m.get("mva") is not None:
        m["mva_ci95"] = [round(x, 4) for x in wilson_ci(m["mva"], m["n_gold"])]
        m["balanced_accuracy_ci95"] = [
            round(x, 4) for x in wilson_ci(m["balanced_accuracy"], m["n_gold"])]
    return m


def evaluate(lang, stage, alpha=None):
    with open(config.SPLITS_DIR / lang / "test_set.json") as f:
        records = json.load(f)
    with open(config.SPLITS_DIR / lang / "cluster_meta.json") as f:
        expected = json.load(f)["expected_yes_rates"]

    label = f"local_{config.LOCAL_MODEL_LABEL}_{stage}_{lang}"
    pred_path = config.PREDICTIONS_DIR / f"{label}_predictions.json"

    if pred_path.exists():
        print(f"[eval] reusing predictions {pred_path}", flush=True)
        with open(pred_path) as f:
            preds = json.load(f)
    else:
        from models.local_agent import system_predictions
        preds = system_predictions(records, adapter_paths(lang, stage))
        config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
        with open(pred_path, "w") as f:
            json.dump(preds, f)
        print(f"[eval] saved predictions {pred_path}", flush=True)

    result = {
        "label": label,
        "language": lang,
        "stage": stage,
        "base_model": config.LOCAL_BASE_MODEL,
        "reward_formula": "(1-alpha)*r_indiv + alpha*r_team",
    }
    if alpha is not None:
        result["alpha"] = alpha
    result.update(core_metrics(preds, records, expected))
    result["all_clusters_subset"] = subset_metrics(
        preds, records, expected, len(config.CLUSTER_KEYS))
    result["two_plus_clusters_subset"] = subset_metrics(preds, records, expected, 2)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"{label}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[eval] {label}: mva={result.get('mva')} "
          f"f1_macro={result.get('f1_macro')} "
          f"bal_acc={result.get('balanced_accuracy')}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--stage", required=True, choices=STAGES)
    args = ap.parse_args()
    alpha = config.GRPO_ALPHA if args.stage.startswith("grpo") else None
    evaluate(args.lang, args.stage, alpha=alpha)


if __name__ == "__main__":
    main()
