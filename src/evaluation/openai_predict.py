#!/usr/bin/env python3
"""
OpenAI 3-Agent Test-Set Prediction + Results
=============================================
Batch-parallel test-set predictions for a fine-tuned OpenAI 3-agent system,
plus metric computation from the saved predictions.

Predictions are saved first (idempotent: skipped when the file already
exists), then all metrics are computed from the saved predictions via the
shared pure-python metric functions in evaluation.evaluate_local, so any
metric can be recomputed later without re-running inference.

Prediction schema (aligned with test-set order), one record per text:
  {"text_id", "agent_predictions": {"cluster1": "YES", ...},
   "agent_p_yes": {"cluster1": 1.0, ...}, "team_vote", "unanimous"}

Usage:
    python3 src/evaluation/openai_predict.py \
        --model-ids artifacts/model_ids/es_sft_41mini_model_ids.json \
        --lang es --label gpt41mini_sft_es

    # predictions only (do not touch artifacts/results/):
    python3 src/evaluation/openai_predict.py \
        --model-ids artifacts/model_ids/sft_41mini_model_ids.json \
        --lang en --label sft_41mini --no-results

API key: read from the OPENAI_API_KEY environment variable, falling back to
parsing the `.env` file at the repo root (line `OPENAI_API_KEY=...`).
The key is never printed and never written to any file.
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from evaluation.evaluate_local import core_metrics, subset_metrics  # noqa: E402

MAX_WORKERS = 16
MAX_RETRIES = 3


def ts():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def load_api_key():
    """OPENAI_API_KEY from env, falling back to `.env` at the repo root."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_path = config.REPO_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith("OPENAI_API_KEY"):
                    _, _, val = line.partition("=")
                    val = val.strip().strip('"').strip("'")
                    if val:
                        return val
    raise SystemExit(
        "OPENAI_API_KEY not found in the environment or in "
        f"{env_path} (expected line: OPENAI_API_KEY=...)"
    )


def load_model_ids(path):
    """Load a model-ids JSON (legacy or neutral keys) -> {cluster1..3: id}."""
    with open(path) as f:
        raw = json.load(f)
    model_ids = config.normalize_cluster_keyed(raw)
    missing = [ck for ck in config.CLUSTER_KEYS if not model_ids.get(ck)]
    if missing:
        raise SystemExit(f"{path} is missing model ids for: {missing}")
    return {ck: model_ids[ck] for ck in config.CLUSTER_KEYS}


def base_model_from_id(model_id):
    """'ft:gpt-4.1-mini-2025-04-14:org:suffix:xyz' -> 'gpt-4.1-mini-2025-04-14'."""
    if model_id.startswith("ft:"):
        parts = model_id.split(":")
        if len(parts) > 1 and parts[1]:
            return parts[1]
    return model_id


def infer_stage(label):
    low = label.lower()
    if "grpo" in low:
        return "grpo_a20" if ("a20" in low or "alpha02" in low) else "grpo"
    if "marspo" in low:
        return "marspo"
    if "sft_pure" in low or "pure" in low:
        return "sft_pure"
    if "dpo" in low:
        return "dpo"
    if "sft" in low:
        return "sft"
    if "base" in low:
        return "base"
    return "unknown"


def clean_tweet(tweet):
    return tweet.replace("\x00", "")


def predict_one(client, model_id, tweet):
    """Single YES/NO prediction with retries (3x, exponential backoff)."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Tweet: {tweet}"},
    ]
    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=3,
                temperature=0,
            )
            raw = (resp.choices[0].message.content or "").strip().upper()
            return "YES" if "YES" in raw else "NO"
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"{ts()}   retry {attempt + 1}/{MAX_RETRIES} "
                  f"after {type(e).__name__}: {e}", flush=True)
            time.sleep(delay)
            delay *= 2


def predict_agent(client, model_id, tweets, agent_key):
    """Predict all tweets for one agent, order-aligned with the input list."""
    print(f"{ts()} {agent_key}: {model_id} ({len(tweets)} texts, "
          f"{MAX_WORKERS} workers)", flush=True)
    preds = [None] * len(tweets)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(predict_one, client, model_id, t): i
                   for i, t in enumerate(tweets)}
        done = 0
        for fut in as_completed(futures):
            preds[futures[fut]] = fut.result()
            done += 1
            if done % 50 == 0 or done == len(tweets):
                print(f"{ts()}   {agent_key}: {done}/{len(tweets)}", flush=True)
    return preds


def build_predictions(model_ids, records):
    """Run all 3 agents and assemble prediction records (test-set order)."""
    client = OpenAI(api_key=load_api_key())
    tweets = [clean_tweet(r["tweet"]) for r in records]

    agent_preds = {}
    for ck in config.CLUSTER_KEYS:
        agent_preds[ck] = predict_agent(client, model_ids[ck], tweets, ck)

    preds = []
    for i, r in enumerate(records):
        ap = {ck: agent_preds[ck][i] for ck in config.CLUSTER_KEYS}
        yes_count = sum(1 for v in ap.values() if v == "YES")
        preds.append({
            "text_id": r["text_id"],
            "agent_predictions": ap,
            "agent_p_yes": {ck: (1.0 if ap[ck] == "YES" else 0.0)
                            for ck in config.CLUSTER_KEYS},
            "team_vote": "YES" if yes_count >= 2 else "NO",
            "unanimous": yes_count in (0, len(config.CLUSTER_KEYS)),
        })
    return preds


def main():
    parser = argparse.ArgumentParser(
        description="OpenAI 3-agent test-set predictions + results")
    parser.add_argument("--model-ids", required=True,
                        help="Path to a *_model_ids.json (legacy or neutral keys)")
    parser.add_argument("--lang", required=True, choices=config.LANGUAGES)
    parser.add_argument("--label", required=True,
                        help="Method key used in output file names")
    parser.add_argument("--results", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Write artifacts/results/<label>_results.json "
                             "(default: yes; use --no-results for predictions only)")
    parser.add_argument("--stage", default=None,
                        help="Stage metadata for the results file "
                             "(default: inferred from --label)")
    args = parser.parse_args()

    model_ids = load_model_ids(args.model_ids)

    test_path = config.SPLITS_DIR / args.lang / "test_set.json"
    with open(test_path) as f:
        records = json.load(f)
    print(f"{ts()} label={args.label} lang={args.lang} "
          f"test_set={test_path} ({len(records)} texts)", flush=True)

    # ── Predictions (idempotent) ──────────────────────────────
    config.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    pred_path = config.PREDICTIONS_DIR / f"{args.label}_predictions.json"
    if pred_path.exists():
        print(f"{ts()} predictions exist, skipping inference: {pred_path}",
              flush=True)
        with open(pred_path) as f:
            preds = json.load(f)
    else:
        preds = build_predictions(model_ids, records)
        with open(pred_path, "w") as f:
            json.dump(preds, f, indent=2)
        print(f"{ts()} saved predictions: {pred_path}", flush=True)

    if len(preds) != len(records):
        raise SystemExit(
            f"{pred_path} has {len(preds)} records but the test set has "
            f"{len(records)}; delete the predictions file and re-run.")

    if not args.results:
        print(f"{ts()} --no-results: done.", flush=True)
        return

    # ── Results ───────────────────────────────────────────────
    with open(config.SPLITS_DIR / args.lang / "cluster_meta.json") as f:
        expected = json.load(f)["expected_yes_rates"]

    result = {
        "label": args.label,
        "language": args.lang,
        "stage": args.stage or infer_stage(args.label),
        "base_model": base_model_from_id(model_ids[config.CLUSTER_KEYS[0]]),
        "model_ids": model_ids,
    }
    result.update(core_metrics(preds, records, expected))
    result["all_clusters_subset"] = subset_metrics(
        preds, records, expected, len(config.CLUSTER_KEYS))
    result["two_plus_clusters_subset"] = subset_metrics(
        preds, records, expected, 2)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.RESULTS_DIR / f"{args.label}_results.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"{ts()} {args.label}: mva={result.get('mva')} "
          f"f1_macro={result.get('f1_macro')} "
          f"bal_acc={result.get('balanced_accuracy')}", flush=True)
    print(f"{ts()} saved results: {out_path}", flush=True)


if __name__ == "__main__":
    main()
