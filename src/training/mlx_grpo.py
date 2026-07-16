#!/usr/bin/env python3
"""
GRPO via rejection sampling for the local Qwen3-8B agents.

Port of the OpenAI-based src/training/grpo_rejection_sampling.py with the
reward reformulated as a convex combination:

    R_j = (1 - alpha) * r_indiv_j + alpha * r_team

r_indiv_j = 1 if the sample matches agent j's own cluster majority on the
text, else 0. r_team = 1 if the team vote (this sample + the other two
agents' majority-of-K labels) matches the overall human majority, else 0.

At alpha = 0.2 with keep-threshold 0.5 this selects the identical sample set
as the legacy formula r_indiv + alpha' * r_team at alpha' = 0.2, so the
existing OpenAI results remain comparable (alpha = alpha' / (1 + alpha')).
Run `--self-test` to verify that equivalence.

Sampling is exact and cheap: each (agent, text) needs one prefill for
p(YES), then K Bernoulli draws at the sampling temperature.

Run under .venv-mlx.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


def compute_reward(agent_label, cluster_key, record, team_vote, alpha):
    """Convex-combination reward for one sampled label."""
    cv = record.get("cluster_votes", {})
    r_indiv = 0.0
    if cluster_key in cv and cv[cluster_key].get("majority") is not None:
        r_indiv = 1.0 if agent_label == cv[cluster_key]["majority"] else 0.0
    r_team = 0.0
    if team_vote is not None and record.get("overall_majority") is not None:
        r_team = 1.0 if team_vote == record["overall_majority"] else 0.0
    return (1 - alpha) * r_indiv + alpha * r_team


def legacy_reward(agent_label, cluster_key, record, team_vote, alpha_prime):
    """Legacy (unnormalized) reward, kept only for the equivalence self-test."""
    cv = record.get("cluster_votes", {})
    r_indiv = 0.0
    if cluster_key in cv and cv[cluster_key].get("majority") is not None:
        r_indiv = 1.0 if agent_label == cv[cluster_key]["majority"] else 0.0
    r_team = 0.0
    if team_vote is not None and record.get("overall_majority") is not None:
        r_team = 1.0 if team_vote == record["overall_majority"] else 0.0
    return r_indiv + alpha_prime * r_team


def self_test():
    """Selection equivalence at alpha=0.2 / threshold 0.5, over every
    (r_indiv, r_team) combination that can occur."""
    thr = config.GRPO_REWARD_THRESHOLD
    rec_yes = {"cluster_votes": {"cluster1": {"majority": "YES"}},
               "overall_majority": "YES"}
    cases = [("YES", "YES"), ("YES", "NO"), ("NO", "YES"), ("NO", "NO")]
    for label, team in cases:
        new = compute_reward(label, "cluster1", rec_yes, team, 0.2)
        old = legacy_reward(label, "cluster1", rec_yes, team, 0.2)
        assert (new > thr) == (old > thr), (label, team, new, old)
    print("self-test passed: alpha=0.2 selection identical to legacy formula")


def majority_label(samples):
    """Majority of an agent's K sampled labels; ties resolve to NO."""
    yes = sum(1 for s in samples if s == "YES")
    return "YES" if yes * 2 > len(samples) else "NO"


def sample_and_filter(lang, alpha, max_texts, k, temperature):
    """Sample K labels per (agent, text), score, and keep reward > threshold.

    Returns {cluster_key: [(tweet, label, reward), ...]} deduplicated by
    (text, label) keeping the highest reward.
    """
    import mlx.core as mx
    from models.local_agent import LocalAgent, clean_text

    with open(config.SPLITS_DIR / lang / "train_set.json") as f:
        records = json.load(f)
    rng = random.Random(config.RANDOM_STATE)
    if max_texts and len(records) > max_texts:
        records = rng.sample(records, max_texts)
    tweets = [clean_text(r["tweet"]) for r in records]

    p_yes = {}
    for ck in config.CLUSTER_KEYS:
        adapter = config.ADAPTERS_DIR / lang / f"sft_{ck}"
        print(f"[mlx_grpo] scoring {len(tweets)} texts with {ck}", flush=True)
        agent = LocalAgent(ck, adapter_path=adapter)
        p_yes[ck] = agent.p_yes_batch(tweets, log_prefix=f"{ck} ")
        del agent
        mx.clear_cache()

    samples = {ck: [LocalAgent.sample_k(p, k=k, temperature=temperature, rng=rng)
                    for p in p_yes[ck]]
               for ck in config.CLUSTER_KEYS}

    kept = {ck: {} for ck in config.CLUSTER_KEYS}
    for ck in config.CLUSTER_KEYS:
        others = [o for o in config.CLUSTER_KEYS if o != ck]
        for i, rec in enumerate(records):
            other_votes = [majority_label(samples[o][i]) for o in others]
            for label in samples[ck][i]:
                votes = other_votes + [label]
                team = "YES" if votes.count("YES") >= 2 else "NO"
                r = compute_reward(label, ck, rec, team, alpha)
                if r > config.GRPO_REWARD_THRESHOLD:
                    key = (i, label)
                    if key not in kept[ck] or r > kept[ck][key][2]:
                        kept[ck][key] = (tweets[i], label, r)
    return {ck: list(v.values()) for ck, v in kept.items()}


def write_grpo_data(lang, alpha_tag, kept):
    """Write kept samples as mlx SFT data (think-prefixed targets)."""
    for ck, rows in kept.items():
        out_dir = config.MLX_DATA_DIR / lang / f"grpo_a{alpha_tag}_{ck}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "train.jsonl", "w") as f:
            for tweet, label, _ in rows:
                f.write(json.dumps({"messages": [
                    {"role": "system", "content": config.SYSTEM_PROMPT},
                    {"role": "user", "content": f"Tweet: {tweet}"},
                    {"role": "assistant",
                     "content": f"<think>\n\n</think>\n\n{label}"},
                ]}) + "\n")
        # mlx_lm.lora requires a valid.jsonl; reuse the SFT validation set.
        shutil.copy(config.MLX_DATA_DIR / lang / f"sft_{ck}" / "valid.jsonl",
                    out_dir / "valid.jsonl")
        print(f"[mlx_grpo] {ck}: kept {len(rows)} samples -> {out_dir}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=config.LANGUAGES)
    ap.add_argument("--alpha", type=float, default=config.GRPO_ALPHA)
    ap.add_argument("--max-texts", type=int, default=1000)
    ap.add_argument("--k", type=int, default=config.GRPO_K_SAMPLES)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--sample-only", action="store_true",
                    help="sample + write data, skip retraining")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.lang:
        ap.error("--lang is required unless --self-test")

    alpha_tag = f"{int(round(args.alpha * 100)):02d}"
    kept = sample_and_filter(args.lang, args.alpha, args.max_texts,
                             args.k, config.GRPO_TEMPERATURE)
    write_grpo_data(args.lang, alpha_tag, kept)
    if args.sample_only:
        return

    from training.mlx_sft import train_sft
    for ck in config.CLUSTER_KEYS:
        sft_adapter = (config.ADAPTERS_DIR / args.lang / f"sft_{ck}"
                       / "adapters.safetensors")
        train_sft(args.lang, ck,
                  data_subdir=f"grpo_a{alpha_tag}_{ck}",
                  adapter_name=f"grpo_a{alpha_tag}_{ck}",
                  resume_adapter=sft_adapter,
                  epochs=1)


if __name__ == "__main__":
    main()
