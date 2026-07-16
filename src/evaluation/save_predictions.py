#!/usr/bin/env python3
"""
save_predictions.py
===================
Run per-text predictions for every fine-tuned method + baselines and dump the
results to ``artifacts/predictions/<method>_predictions.json``.  The aggregate
result JSONs in ``artifacts/results/`` are left untouched.

Why: per-cluster F1 requires per-text predictions, which the original evaluator
discards after computing aggregate stats.  This script captures them so every
downstream F1 plot/table has exact values instead of analytical approximations.

Usage:
    export OPENAI_API_KEY=sk-...
    python3 src/evaluation/save_predictions.py            # run every method
    python3 src/evaluation/save_predictions.py --only grpo_41mini_alpha02
    python3 src/evaluation/save_predictions.py --overwrite

Cost: ~1,026 calls per multi-agent method × ~$0.00075/call ≈ $0.77/method.
Baselines (always_no, zero_shot_ensemble, persona_prompts) add one-agent or
three-agent runs on top; total ~$5 for every method listed below.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.agent import Agent, MultiAgentSystem

MODELS_DIR = "artifacts/model_ids"
PREDICTIONS_DIR = "artifacts/predictions"
TEST_SET_PATH = str(config.SPLITS_DIR / "test_set.json")

# Every method we want per-text predictions for.  Baselines are special-cased
# because they don't have a model_ids file.
FINE_TUNED_METHODS = [
    "sft_41mini", "sft_pure_41mini",
    "dpo_41mini", "dpo_bb03_41mini", "dpo_bb05_41mini",
    "marspo_41mini",
    "grpo_41mini_alpha00", "grpo_41mini_alpha02",
    "grpo_41mini_alpha05", "grpo_41mini_alpha10",
    "sft",    # gpt-4o-mini SFT (initial study)
    "grpo",   # gpt-4o-mini GRPO
]
BASELINES = ["always_no", "zero_shot_ensemble", "persona_prompts"]


def load_test_set():
    with open(TEST_SET_PATH) as f:
        return json.load(f)


def save(key, records):
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    path = os.path.join(PREDICTIONS_DIR, f"{key}_predictions.json")
    with open(path, "w") as f:
        json.dump(records, f)
    print(f"  Saved {len(records)} predictions -> {path}")


def run_multi_agent(key, model_ids, test_data, delay=0.05):
    print(f"\n[{key}] multi-agent ({len(test_data)} texts, 3 agents)")
    system = MultiAgentSystem(model_ids=model_ids)
    tweets = [t["tweet"] for t in test_data]
    results = system.predict_batch(tweets, delay=delay)
    records = [
        {
            "text_id": t.get("text_id"),
            "agent_predictions": r["agent_predictions"],
            "team_vote": r["team_vote"],
            "unanimous": r.get("unanimous"),
        }
        for r, t in zip(results, test_data)
    ]
    save(key, records)


def run_always_no(test_data):
    print("\n[always_no] trivial baseline (no API calls)")
    records = [
        {
            "text_id": t.get("text_id"),
            "agent_predictions": {c: "NO" for c in config.CLUSTER_NAMES},
            "team_vote": "NO",
            "unanimous": True,
        }
        for t in test_data
    ]
    save("always_no", records)


def run_zero_shot_ensemble(test_data, delay=0.05):
    print("\n[zero_shot_ensemble] three copies of base model")
    agent = Agent("single", config.BASE_MODEL)
    preds = [r["label"] for r in agent.predict_batch(
        [t["tweet"] for t in test_data], delay=delay)]
    records = [
        {
            "text_id": t.get("text_id"),
            "agent_predictions": {c: p for c in config.CLUSTER_NAMES},
            "team_vote": p,
            "unanimous": True,
        }
        for p, t in zip(preds, test_data)
    ]
    save("zero_shot_ensemble", records)


def run_persona_prompts(test_data, delay=0.05):
    from openai import OpenAI
    prompts = {
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
    print(f"\n[persona_prompts] three persona prompts ({len(test_data)} texts each)")
    all_preds = {}
    for cluster, prompt in prompts.items():
        ps = []
        for t in test_data:
            resp = client.chat.completions.create(
                model=config.BASE_MODEL,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Tweet: {t['tweet']}"},
                ],
                temperature=0.0,
                max_tokens=3,
            )
            raw = resp.choices[0].message.content.strip().upper()
            ps.append("YES" if "YES" in raw else "NO")
            time.sleep(delay)
        all_preds[cluster] = ps
    records = []
    for i, t in enumerate(test_data):
        votes = {c: all_preds[c][i] for c in config.CLUSTER_NAMES}
        yes = sum(1 for v in votes.values() if v == "YES")
        records.append({
            "text_id": t.get("text_id"),
            "agent_predictions": votes,
            "team_vote": "YES" if yes >= 2 else "NO",
            "unanimous": yes in (0, 3),
        })
    save("persona_prompts", records)


def main():
    parser = argparse.ArgumentParser(description="Save per-text predictions.")
    parser.add_argument("--only", help="Run only this method key.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-run even if <key>_predictions.json exists.")
    args = parser.parse_args()

    test_data = load_test_set()
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    todo = BASELINES + FINE_TUNED_METHODS
    if args.only:
        todo = [args.only]

    for key in todo:
        out = os.path.join(PREDICTIONS_DIR, f"{key}_predictions.json")
        if os.path.exists(out) and not args.overwrite:
            print(f"[{key}] skip (exists)")
            continue

        if key == "always_no":
            run_always_no(test_data)
        elif key == "zero_shot_ensemble":
            run_zero_shot_ensemble(test_data)
        elif key == "persona_prompts":
            run_persona_prompts(test_data)
        else:
            ids_path = os.path.join(MODELS_DIR, f"{key}_model_ids.json")
            if not os.path.exists(ids_path):
                print(f"[{key}] skip (no {ids_path})")
                continue
            with open(ids_path) as f:
                model_ids = json.load(f)
            run_multi_agent(key, model_ids, test_data)

    print("\nDone. Predictions in artifacts/predictions/.")
    print("All F1 helpers in report_assets/utils.py pick these up automatically.")


if __name__ == "__main__":
    main()
