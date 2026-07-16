#!/usr/bin/env python3
"""
One-run orchestrator for the bilingual local (Qwen3-8B, MLX) experiments.

Runs every step end-to-end for both languages: clustering -> ablation ->
data prep -> mlx formatting -> base eval -> SFT -> DPO -> GRPO -> evals ->
balanced-SFT ablation -> behavioral fidelity. Restart-safe: each step is
skipped iff the state file marks it done AND its output files exist.

Usage (from repo root):
    .venv-mlx/bin/python src/pipeline/run_local_experiments.py
    ... --lang en            # one language only
    ... --until eval_sft_en  # stop after a step
    ... --force sft_en_cluster1  # re-run one step (or "all")

Logs: artifacts/logs/<step>.log (+ artifacts/logs/local_pipeline.log).
State: artifacts/local_state.json
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

STATE_PATH = config.ARTIFACTS_DIR / "local_state.json"
PY = config.MLX_PYTHON
SRC = config.REPO_ROOT / "src"


def step_list(langs):
    """Ordered [(name, cmd, outputs)] for the requested languages."""
    steps = []

    def add(name, cmd, outputs):
        steps.append((name, cmd, [Path(o) for o in outputs]))

    add("env_check",
        [PY, "-c", "import mlx_lm, pandas, sklearn, scipy; print('env ok')"],
        [])

    for lang in langs:
        add(f"clustering_{lang}",
            [PY, str(SRC / "preprocessing/behavioral_clustering.py"),
             "--lang", lang],
            [config.DATA_DIR / f"behavioral_features_{lang}.csv"])
        add(f"ablation_{lang}",
            [PY, str(SRC / "preprocessing/clustering_ablation.py"),
             "--lang", lang],
            [config.RESULTS_DIR / f"clustering_ablation_{lang}.json"])
        prep_cmd = [PY, str(SRC / "preprocessing/prepare_preference_data.py"),
                    "--lang", lang]
        prep_out = [config.SPLITS_DIR / lang / "test_set.json",
                    config.SPLITS_DIR / lang / "cluster_meta.json"]
        if lang == "en":
            prep_cmd.append("--balance-labels")
            prep_out.append(config.PREFERENCE_DIR / lang
                            / "sft_balanced_cluster1_train.jsonl")
        add(f"prep_{lang}", prep_cmd, prep_out)
        add(f"format_mlx_{lang}",
            [PY, str(SRC / "preprocessing/format_mlx_data.py"), "--lang", lang],
            [config.MLX_DATA_DIR / lang / "sft_cluster1" / "train.jsonl"])

        add(f"eval_base_{lang}",
            [PY, str(SRC / "evaluation/evaluate_local.py"),
             "--lang", lang, "--stage", "base"],
            [config.RESULTS_DIR
             / f"local_{config.LOCAL_MODEL_LABEL}_base_{lang}_results.json"])

        for ck in config.CLUSTER_KEYS:
            add(f"sft_{lang}_{ck}",
                [PY, str(SRC / "training/mlx_sft.py"),
                 "--lang", lang, "--cluster", ck],
                [config.ADAPTERS_DIR / lang / f"sft_{ck}" / "adapters.safetensors"])
        add(f"eval_sft_{lang}",
            [PY, str(SRC / "evaluation/evaluate_local.py"),
             "--lang", lang, "--stage", "sft"],
            [config.RESULTS_DIR
             / f"local_{config.LOCAL_MODEL_LABEL}_sft_{lang}_results.json"])

        for ck in config.CLUSTER_KEYS:
            add(f"dpo_{lang}_{ck}",
                [PY, str(SRC / "training/mlx_dpo.py"),
                 "--lang", lang, "--cluster", ck],
                [config.ADAPTERS_DIR / lang / f"dpo_{ck}" / "adapters.safetensors"])
        add(f"eval_dpo_{lang}",
            [PY, str(SRC / "evaluation/evaluate_local.py"),
             "--lang", lang, "--stage", "dpo"],
            [config.RESULTS_DIR
             / f"local_{config.LOCAL_MODEL_LABEL}_dpo_{lang}_results.json"])

        add(f"grpo_{lang}",
            [PY, str(SRC / "training/mlx_grpo.py"), "--lang", lang],
            [config.ADAPTERS_DIR / lang / "grpo_a20_cluster1"
             / "adapters.safetensors"])
        add(f"eval_grpo_{lang}",
            [PY, str(SRC / "evaluation/evaluate_local.py"),
             "--lang", lang, "--stage", "grpo_a20"],
            [config.RESULTS_DIR
             / f"local_{config.LOCAL_MODEL_LABEL}_grpo_a20_{lang}_results.json"])

    if "en" in langs:
        for ck in config.CLUSTER_KEYS:
            add(f"sft_balanced_en_{ck}",
                [PY, str(SRC / "training/mlx_sft.py"),
                 "--lang", "en", "--cluster", ck, "--balanced"],
                [config.ADAPTERS_DIR / "en" / f"sft_balanced_{ck}"
                 / "adapters.safetensors"])
        add("eval_sft_balanced_en",
            [PY, str(SRC / "evaluation/evaluate_local.py"),
             "--lang", "en", "--stage", "sft_balanced"],
            [config.RESULTS_DIR
             / f"local_{config.LOCAL_MODEL_LABEL}_sft_balanced_en_results.json"])

    for lang in langs:
        add(f"fidelity_{lang}",
            [PY, str(SRC / "evaluation/behavioral_fidelity.py"),
             "--lang", lang, "--stages", "base", "sft", "dpo", "grpo_a20"],
            [config.RESULTS_DIR / f"behavioral_fidelity_{lang}.json"])

    return steps


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=config.LANGUAGES, default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--force", default=None,
                    help="step name to re-run, or 'all'")
    args = ap.parse_args()

    langs = [args.lang] if args.lang else config.LANGUAGES
    steps = step_list(langs)
    state = {} if args.force == "all" else load_state()
    if args.force and args.force != "all":
        state.pop(args.force, None)

    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    main_log = open(config.LOGS_DIR / "local_pipeline.log", "a")

    def log(msg):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line, flush=True)
        main_log.write(line + "\n")
        main_log.flush()

    log(f"=== run start: langs={langs} until={args.until} ===")
    for name, cmd, outputs in steps:
        done = (state.get(name, {}).get("status") == "done"
                and all(o.exists() for o in outputs))
        if done:
            log(f"skip {name} (done)")
        else:
            log(f"run  {name}")
            t0 = time.time()
            with open(config.LOGS_DIR / f"{name}.log", "w") as step_log:
                proc = subprocess.run(cmd, stdout=step_log,
                                      stderr=subprocess.STDOUT,
                                      cwd=config.REPO_ROOT)
            dt = time.time() - t0
            if proc.returncode != 0:
                log(f"FAIL {name} (exit {proc.returncode}, {dt:.0f}s) — "
                    f"see artifacts/logs/{name}.log")
                sys.exit(1)
            missing = [str(o) for o in outputs if not o.exists()]
            if missing:
                log(f"FAIL {name}: expected outputs missing: {missing}")
                sys.exit(1)
            state[name] = {"status": "done", "seconds": round(dt, 1),
                           "finished_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            save_state(state)
            log(f"done {name} ({dt:.0f}s)")
        if args.until and name == args.until:
            log(f"=== stopped at --until {args.until} ===")
            return
    log("=== all steps complete ===")


if __name__ == "__main__":
    main()
