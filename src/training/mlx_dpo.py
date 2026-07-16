#!/usr/bin/env python3
"""
DPO for the local Qwen3-8B agents, implemented directly in MLX.

Because every completion is a single YES/NO token, the DPO loss is exact and
cheap: log p(y|x) is one token's log-probability at the last prompt position,
and one forward pass yields both the chosen and rejected log-probs (same
prompt, two candidate tokens). Loss per pair:

    -log sigmoid( beta * [ (lp_pol(w) - lp_ref(w)) - (lp_pol(l) - lp_ref(l)) ] )

The reference model is the SFT policy (matching the OpenAI DPO setup, which
fine-tunes from the SFT model). Reference log-probs are computed once with
the freshly loaded SFT weights before any optimizer step, so only one model
is ever in memory.

Fallback if this trainer misbehaves: the community package `mlx-lm-lora`
ships a DPO trainer that can cross-check results.

Run under .venv-mlx.
"""
import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DPO_LR = 1e-5
DPO_EPOCHS = 2
DPO_BATCH = 8


def load_pairs(lang, cluster_key, tokenizer):
    """Tokenized DPO pairs: (prompt_tokens, chosen_tok_id, rejected_tok_id)."""
    path = config.MLX_DATA_DIR / lang / f"dpo_{cluster_key}.jsonl"
    pairs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            text = tokenizer.apply_chat_template(
                row["prompt_messages"], add_generation_prompt=True,
                tokenize=False, enable_thinking=False)
            toks = tokenizer.encode(text)
            tok_c = tokenizer.encode(row["chosen"], add_special_tokens=False)[0]
            tok_r = tokenizer.encode(row["rejected"], add_special_tokens=False)[0]
            pairs.append((toks, tok_c, tok_r))
    return pairs


def batch_logprobs(model, batch):
    """Log-probs of (chosen, rejected) answer tokens for a batch of pairs.

    Right padding is safe: with a causal mask, positions before the padding
    are unaffected, and we read logits at each row's own last real position.
    """
    maxlen = max(len(t) for t, _, _ in batch)
    inputs = [t + [0] * (maxlen - len(t)) for t, _, _ in batch]
    positions = [len(t) - 1 for t, _, _ in batch]
    logits = model(mx.array(inputs))          # [B, T, V]
    rows = logits[mx.arange(len(batch)), mx.array(positions)]  # [B, V]
    ls = rows - mx.logsumexp(rows, axis=-1, keepdims=True)
    lp_c = ls[mx.arange(len(batch)), mx.array([c for _, c, _ in batch])]
    lp_r = ls[mx.arange(len(batch)), mx.array([r for _, _, r in batch])]
    return lp_c, lp_r


def train_dpo(lang, cluster_key, beta=None, epochs=DPO_EPOCHS):
    beta = beta or config.DPO_BETA
    sft_dir = config.ADAPTERS_DIR / lang / f"sft_{cluster_key}"
    out_dir = config.ADAPTERS_DIR / lang / f"dpo_{cluster_key}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[mlx_dpo] {lang}/{cluster_key}: loading SFT policy", flush=True)
    model, tokenizer = load(config.LOCAL_BASE_MODEL, adapter_path=str(sft_dir))

    # mlx_lm.load applies the LoRA layers but does NOT freeze the base model,
    # so without this, value_and_grad differentiates all 8B parameters and
    # Adam allocates state for them (memory blow-up, ~100x slowdown).
    model.freeze()
    n_trainable = 0
    for _, module in model.named_modules():
        if hasattr(module, "lora_a") and hasattr(module, "lora_b"):
            module.unfreeze(keys=["lora_a", "lora_b"])
            n_trainable += module.lora_a.size + module.lora_b.size
    print(f"[mlx_dpo] trainable LoRA params: {n_trainable:,}", flush=True)
    assert 0 < n_trainable < 100_000_000, \
        f"unexpected trainable-parameter count: {n_trainable}"

    pairs = load_pairs(lang, cluster_key, tokenizer)
    print(f"[mlx_dpo] {len(pairs)} pairs, beta={beta}", flush=True)

    # Reference log-probs: SFT policy before any update.
    ref_c, ref_r = [], []
    for i in range(0, len(pairs), DPO_BATCH):
        lp_c, lp_r = batch_logprobs(model, pairs[i:i + DPO_BATCH])
        mx.eval(lp_c, lp_r)
        ref_c += lp_c.tolist()
        ref_r += lp_r.tolist()
    pre_margin = sum(c - r for c, r in zip(ref_c, ref_r)) / len(pairs)
    print(f"[mlx_dpo] pre-training mean log-margin: {pre_margin:.4f}", flush=True)

    def loss_fn(model, batch, batch_ref_c, batch_ref_r):
        lp_c, lp_r = batch_logprobs(model, batch)
        logits = beta * ((lp_c - batch_ref_c) - (lp_r - batch_ref_r))
        return -nn.log_sigmoid(logits).mean()

    value_and_grad = nn.value_and_grad(model, loss_fn)
    opt = optim.Adam(learning_rate=DPO_LR)
    rng = random.Random(config.RANDOM_STATE)

    # Length-bucketed batches: batch neighbors in prompt length so padding
    # (and the [B, T, V] logits peak) stays small, then shuffle at the batch
    # level. Prevents the memory spikes that OOM-killed long-prompt batches.
    by_len = sorted(range(len(pairs)), key=lambda j: len(pairs[j][0]))
    batches = [by_len[i:i + DPO_BATCH] for i in range(0, len(by_len), DPO_BATCH)]

    step = 0
    for epoch in range(epochs):
        rng.shuffle(batches)
        for idx in batches:
            batch = [pairs[j] for j in idx]
            brc = mx.array([ref_c[j] for j in idx])
            brr = mx.array([ref_r[j] for j in idx])
            loss, grads = value_and_grad(model, batch, brc, brr)
            opt.update(model, grads)
            mx.eval(model.parameters(), opt.state)
            step += 1
            if step % 10 == 0:
                mx.clear_cache()
            if step % 20 == 0:
                print(f"  epoch {epoch + 1} step {step} loss {loss.item():.4f}",
                      flush=True)

    # Sanity: the policy must now prefer the chosen label more than before.
    post_c, post_r = [], []
    for i in range(0, len(pairs), DPO_BATCH):
        lp_c, lp_r = batch_logprobs(model, pairs[i:i + DPO_BATCH])
        mx.eval(lp_c, lp_r)
        post_c += lp_c.tolist()
        post_r += lp_r.tolist()
    post_margin = sum(c - r for c, r in zip(post_c, post_r)) / len(pairs)
    print(f"[mlx_dpo] post-training mean log-margin: {post_margin:.4f}", flush=True)
    if post_margin <= pre_margin:
        raise RuntimeError(
            f"DPO sanity check failed: margin {pre_margin:.4f} -> "
            f"{post_margin:.4f} did not improve")

    weights = dict(tree_flatten(model.trainable_parameters()))
    mx.save_safetensors(str(out_dir / "adapters.safetensors"), weights)
    shutil.copy(sft_dir / "adapter_config.json", out_dir / "adapter_config.json")
    print(f"[mlx_dpo] saved {out_dir}", flush=True)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--beta", type=float, default=None)
    args = ap.parse_args()
    clusters = [args.cluster] if args.cluster else config.CLUSTER_KEYS
    for ck in clusters:
        train_dpo(args.lang, ck, beta=args.beta)
        mx.clear_cache()


if __name__ == "__main__":
    main()
