#!/usr/bin/env python3
"""
MLX Data Formatter
===================
Converts the OpenAI-format preference data under data/preference/{lang}/
into mlx-lm-compatible datasets under data/mlx/{lang}/.

SFT: data/preference/{lang}/sft_*.jsonl (OpenAI chat format) ->
     data/mlx/{lang}/<dataset>/{train,valid}.jsonl in mlx-lm chat format
     (same {"messages": [...]} envelope -- mlx_lm.lora accepts "messages"
     JSONL directly), with the assistant content rewritten to
     "<think>\\n\\n</think>\\n\\n" + label so training targets match Qwen3's
     enable_thinking=False generation continuation.
     `format_sft_file(src_path, dst_dir)` is reusable (e.g. for GRPO
     rejection-sampling files produced later).

DPO: data/preference/{lang}/dpo_cluster{i}_{train,val}.jsonl (OpenAI DPO
     format {"input": {"messages": [...]}, "preferred_output": [...],
     "non_preferred_output": [...]}) ->
     data/mlx/{lang}/dpo_cluster{i}.jsonl        (train pairs)
     data/mlx/{lang}/dpo_cluster{i}_valid.jsonl  (val pairs)
     as {"prompt_messages": [...], "chosen": "YES"/"NO", "rejected": ...}.

Usage:
  python3 src/preprocessing/format_mlx_data.py --lang en|es
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

THINK_PREFIX = "<think>\n\n</think>\n\n"


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(items, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def _split_name(stem):
    """Map a source-file stem to the mlx split filename."""
    if stem.endswith("_train"):
        return "train"
    if stem.endswith("_val") or stem.endswith("_valid"):
        return "valid"
    return "train"


def _dataset_name(stem):
    """Strip the split suffix from a source-file stem."""
    for suffix in ("_train", "_valid", "_val"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def format_sft_file(src_path, dst_dir, split_name=None):
    """Convert one OpenAI chat-format SFT JSONL to mlx-lm chat format.

    Writes <dst_dir>/<split>.jsonl (split inferred from the source filename
    unless split_name is given). Assistant content becomes
    THINK_PREFIX + label. Returns (dst_path, n_examples).
    """
    src_path = Path(src_path)
    split = split_name or _split_name(src_path.stem)
    out = []
    for ex in _read_jsonl(src_path):
        messages = [dict(m) for m in ex["messages"]]
        assert messages[-1]["role"] == "assistant", f"bad example in {src_path}"
        messages[-1]["content"] = THINK_PREFIX + messages[-1]["content"]
        out.append({"messages": messages})
    dst_path = Path(dst_dir) / f"{split}.jsonl"
    _write_jsonl(out, dst_path)
    return dst_path, len(out)


def convert_dpo_file(src_path, dst_path):
    """Convert one OpenAI DPO-pair JSONL to mlx-friendly pair format.

    Handles both the dict envelope {"input": {"messages": [...]}} and the
    bare-list variant {"input": [...]}. Returns (dst_path, n_pairs).
    """
    out = []
    for pair in _read_jsonl(src_path):
        inp = pair["input"]
        prompt_messages = inp["messages"] if isinstance(inp, dict) else inp
        out.append({
            "prompt_messages": prompt_messages,
            "chosen": pair["preferred_output"][0]["content"],
            "rejected": pair["non_preferred_output"][0]["content"],
        })
    _write_jsonl(out, dst_path)
    return dst_path, len(out)


def main(lang):
    print(f"=== MLX data formatting [{lang}] ===")
    src_dir = config.PREFERENCE_DIR / lang
    dst_root = config.MLX_DATA_DIR / lang
    if not src_dir.exists():
        raise FileNotFoundError(
            f"{src_dir} missing -- run prepare_preference_data.py --lang {lang} first")

    # ── SFT (incl. sft_balanced_*) ──
    sft_files = sorted(src_dir.glob("sft_*.jsonl"))
    for src in sft_files:
        dataset = _dataset_name(src.stem)
        dst_path, n = format_sft_file(src, dst_root / dataset)
        print(f"   {src.name} -> {dst_path.relative_to(config.DATA_DIR)} ({n} examples)")

    # ── DPO pairs ──
    for src in sorted(src_dir.glob("dpo_*.jsonl")):
        dataset = _dataset_name(src.stem)
        split = _split_name(src.stem)
        suffix = "" if split == "train" else "_valid"
        dst_path, n = convert_dpo_file(src, dst_root / f"{dataset}{suffix}.jsonl")
        print(f"   {src.name} -> {dst_path.relative_to(config.DATA_DIR)} ({n} pairs)")

    print(f"Done. Output root: {dst_root}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    main(ap.parse_args().lang)
