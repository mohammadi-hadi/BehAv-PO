#!/usr/bin/env python3
"""
Local MLX agent for the bilingual BehAv-PO experiments (Qwen3-8B).

Every completion in this task is a single YES/NO answer, so inference never
generates autoregressively: one prefill pass gives the next-token logits, and
the answer distribution is read off the first tokens of "YES" and "NO".
The chat template is rendered with enable_thinking=False, which ends the
prompt with an empty "<think>\n\n</think>\n\n" block, so the next token is
the actual answer token.

Run under .venv-mlx (python 3.12 + mlx-lm); not importable from system python.
"""
import math
import random
import sys
from pathlib import Path

import mlx.core as mx
from mlx_lm import load

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402


class LocalAgent:
    """One cluster agent backed by Qwen3-8B (+ optional LoRA adapter)."""

    def __init__(self, cluster_key, adapter_path=None, model_path=None,
                 system_prompt=None):
        self.cluster_key = cluster_key
        self.model_path = model_path or config.LOCAL_BASE_MODEL
        self.adapter_path = str(adapter_path) if adapter_path else None
        self.system_prompt = system_prompt or config.SYSTEM_PROMPT
        self.model, self.tokenizer = load(self.model_path,
                                          adapter_path=self.adapter_path)
        ids_yes = self.tokenizer.encode("YES", add_special_tokens=False)
        ids_no = self.tokenizer.encode("NO", add_special_tokens=False)
        self.tok_yes, self.tok_no = ids_yes[0], ids_no[0]
        assert self.tok_yes != self.tok_no, "YES/NO must differ at first token"

    def _prompt_tokens(self, tweet):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Tweet: {tweet}"},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
            enable_thinking=False)
        return self.tokenizer.encode(text)

    def p_yes(self, tweet):
        """P(YES) from one prefill pass, softmax restricted to {YES, NO}."""
        toks = self._prompt_tokens(tweet)
        logits = self.model(mx.array([toks]))[0, -1]
        ly = logits[self.tok_yes].item()
        ln = logits[self.tok_no].item()
        return 1.0 / (1.0 + math.exp(ln - ly))

    def p_yes_batch(self, tweets, log_every=100, log_prefix=""):
        out = []
        for i, t in enumerate(tweets):
            out.append(self.p_yes(clean_text(t)))
            if log_every and (i + 1) % log_every == 0:
                print(f"  {log_prefix}{i + 1}/{len(tweets)}", flush=True)
        return out

    @staticmethod
    def label(p):
        return "YES" if p >= 0.5 else "NO"

    @staticmethod
    def sample_k(p, k=None, temperature=None, rng=None):
        """K Bernoulli draws of the answer token at the given temperature.

        Exact temperature sampling restricted to {YES, NO}: the two-way
        softmax at temperature tau has P(YES) = sigma(logit(p)/tau).
        """
        k = k or config.GRPO_K_SAMPLES
        temperature = temperature or config.GRPO_TEMPERATURE
        rng = rng or random
        p = min(max(p, 1e-9), 1 - 1e-9)
        logit = math.log(p / (1 - p))
        p_t = 1.0 / (1.0 + math.exp(-logit / temperature))
        return ["YES" if rng.random() < p_t else "NO" for _ in range(k)]


def clean_text(text):
    """Strip null bytes and normalize whitespace (mirrors the OpenAI path)."""
    return text.replace("\x00", " ").strip()


def system_predictions(test_records, adapter_paths, system_prompt=None):
    """Run the 3-agent system over test records, one adapter at a time so
    peak memory stays at a single 8B model.

    adapter_paths: {cluster_key: path-or-None}. Returns a list of prediction
    dicts aligned with test_records.
    """
    tweets = [r["tweet"] for r in test_records]
    per_agent = {}
    for ck in config.CLUSTER_KEYS:
        print(f"[eval] agent {ck} adapter={adapter_paths.get(ck)}", flush=True)
        agent = LocalAgent(ck, adapter_path=adapter_paths.get(ck),
                           system_prompt=system_prompt)
        per_agent[ck] = agent.p_yes_batch(tweets, log_prefix=f"{ck} ")
        del agent
        mx.clear_cache()

    preds = []
    for i, rec in enumerate(test_records):
        labels = {ck: LocalAgent.label(per_agent[ck][i])
                  for ck in config.CLUSTER_KEYS}
        p_yes = {ck: round(per_agent[ck][i], 6) for ck in config.CLUSTER_KEYS}
        yes_count = sum(1 for v in labels.values() if v == "YES")
        preds.append({
            "text_id": rec.get("text_id"),
            "agent_predictions": labels,
            "agent_p_yes": p_yes,
            "team_vote": "YES" if yes_count >= 2 else "NO",
            "unanimous": len(set(labels.values())) == 1,
        })
    return preds
