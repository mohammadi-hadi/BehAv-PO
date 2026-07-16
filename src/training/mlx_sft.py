#!/usr/bin/env python3
"""
LoRA SFT for the local Qwen3-8B agents via mlx_lm.lora (subprocess).

One adapter per cluster per language. Training data comes from
data/mlx/{lang}/sft_cluster{i}/ (written by format_mlx_data.py), where the
assistant targets are "<think>\n\n</think>\n\nYES|NO" to match the
enable_thinking=False generation continuation.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

SFT_BATCH_SIZE = 8
SFT_NUM_LAYERS = 16
SFT_MAX_SEQ = 512


def n_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def train_sft(lang, cluster_key, data_subdir=None, adapter_name=None,
              resume_adapter=None, epochs=None, extra_args=None):
    """Run one mlx_lm.lora job. Returns the adapter directory path."""
    data_dir = config.MLX_DATA_DIR / lang / (data_subdir or f"sft_{cluster_key}")
    adapter_dir = config.ADAPTERS_DIR / lang / (adapter_name or f"sft_{cluster_key}")
    adapter_dir.mkdir(parents=True, exist_ok=True)

    n_train = n_lines(data_dir / "train.jsonl")
    epochs = epochs or config.SFT_EPOCHS
    iters = max(50, (n_train * epochs) // SFT_BATCH_SIZE)

    cmd = [
        config.MLX_PYTHON, "-m", "mlx_lm", "lora",
        "--model", config.LOCAL_BASE_MODEL,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--mask-prompt",
        "--batch-size", str(SFT_BATCH_SIZE),
        "--num-layers", str(SFT_NUM_LAYERS),
        "--max-seq-length", str(SFT_MAX_SEQ),
        "--iters", str(iters),
        "--steps-per-report", "50",
        "--steps-per-eval", "200",
        "--save-every", str(iters),
        "--adapter-path", str(adapter_dir),
        "--seed", str(config.RANDOM_STATE),
    ]
    if resume_adapter:
        cmd += ["--resume-adapter-file", str(resume_adapter)]
    if extra_args:
        cmd += list(extra_args)

    print(f"[mlx_sft] {lang}/{cluster_key}: {n_train} examples, "
          f"{iters} iters -> {adapter_dir}", flush=True)
    subprocess.run(cmd, check=True)
    assert (adapter_dir / "adapters.safetensors").exists(), \
        f"adapter not written: {adapter_dir}"
    return adapter_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=config.LANGUAGES)
    ap.add_argument("--cluster", default=None,
                    help="single cluster key (default: all three)")
    ap.add_argument("--balanced", action="store_true",
                    help="train on the label-balanced SFT ablation data")
    args = ap.parse_args()

    clusters = [args.cluster] if args.cluster else config.CLUSTER_KEYS
    for ck in clusters:
        if args.balanced:
            train_sft(args.lang, ck, data_subdir=f"sft_balanced_{ck}",
                      adapter_name=f"sft_balanced_{ck}")
        else:
            train_sft(args.lang, ck)


if __name__ == "__main__":
    main()
