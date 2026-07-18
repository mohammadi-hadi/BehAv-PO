# BehAv-PO: Full Pipeline Guide

Complete step-by-step guide for running the multi-agent behavioral preference
optimization pipeline for sexism detection on EXIST 2024, in four settings:
GPT-EN, GPT-ES (`gpt-4.1-mini` via the OpenAI API) and Qwen-EN, Qwen-ES
(Qwen3-8B, local MLX LoRA).

All commands are run from the **repository root**. See `README.md` for a
directory map.

**Cluster naming.** Clusters use neutral keys `cluster1`/`cluster2`/`cluster3`,
ordered by ascending YES rate. Display names live in
`src/config.py:CLUSTER_DISPLAY` (single rename point). The frozen English
OpenAI artifacts still contain the legacy names; `config.normalize_*` helpers
rename them at load time.

## Overview

```
data/raw/EXIST2024_training.json
        |
        v
[1. Clustering, per lang]   ->  data/behavioral_features_{en,es}.csv
        |
        v
[2. Data Prep, per lang]    ->  data/preference/{lang}/*.jsonl + data/splits/{lang}/*.json
        |                       (+ data/mlx/{lang}/ for the local stack)
        v
[3. Phase 1: SFT]           ->  3 fine-tuned models per base (one per cluster)
        |
        v
[4. Phase 2: DPO / Mars-PO] ->  3 DPO-refined models
        |
        v
[5. Phase 3: GRPO]          ->  3 GRPO-refined models (convex team reward)
        |
        v
[6. Evaluation]             ->  artifacts/results/*.json + artifacts/predictions/*.json
```

## Prerequisites

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"          # OpenAI settings only

# local (Qwen/MLX) settings, Apple Silicon:
python3 -m venv .venv-mlx && .venv-mlx/bin/pip install -r requirements.txt mlx-lm
```

## Local pipeline in one command (Qwen-EN + Qwen-ES)

The bilingual local suite (clustering, ablation, data prep, MLX formatting,
base eval, SFT, DPO, GRPO, balanced-SFT ablation, behavioral fidelity) runs
end-to-end with:

```bash
.venv-mlx/bin/python src/pipeline/run_local_experiments.py             # both languages
.venv-mlx/bin/python src/pipeline/run_local_experiments.py --lang en   # one language
.venv-mlx/bin/python src/pipeline/run_local_experiments.py --until eval_sft_en
.venv-mlx/bin/python src/pipeline/run_local_experiments.py --force sft_en_cluster1
```

It is restart-safe: state in `artifacts/local_state.json`, per-step logs in
`artifacts/logs/`. Steps 1–2 below are the same scripts the runner calls;
they can also be run manually.

## Step 1: Clustering (per language)

Clusters annotators into 3 behavioral groups (YES rate, agreement rate,
label entropy), separately per language.

```bash
python src/preprocessing/behavioral_clustering.py --lang en
python src/preprocessing/behavioral_clustering.py --lang es   # forces k=3 (see below)
python src/preprocessing/clustering_ablation.py --lang en     # 2-feature ablation (report-only)
python src/preprocessing/clustering_ablation.py --lang es
```

**Input:** `data/raw/EXIST2024_training.json`
**Output:** `data/behavioral_features_{lang}.csv` (348 EN / 390 ES rows, `cluster`
column with values `cluster1..3`), plus `artifacts/clustering/{lang}/` scan
reports and `artifacts/results/clustering_ablation_{lang}.json`.

| Cluster | EN size | EN YES rate | ES size | ES YES rate |
|---------|---------|-------------|---------|-------------|
| cluster1 | 75 | ~0.215 | 67 | ~0.268 |
| cluster2 | 224 | ~0.438 | 234 | ~0.472 |
| cluster3 | 49 | ~0.635 | 89 | ~0.676 |

On Spanish the silhouette criterion prefers k=2 (0.465 vs 0.349 at k=3);
`config.FORCE_K` forces k=3 for architecture parity and the honest scan is
kept in `artifacts/clustering/es/kmeans_scan.json`.

The legacy single-language script
(`src/preprocessing/annotator_clustering_pipeline.py`) produced the frozen
`data/behavioral_features.csv` used by the original GPT-EN runs.

## Step 2: Data Preparation (per language)

```bash
python src/preprocessing/prepare_preference_data.py --lang en
python src/preprocessing/prepare_preference_data.py --lang es
python src/preprocessing/prepare_preference_data.py --lang en --balance-labels  # balanced-SFT ablation
python src/preprocessing/format_mlx_data.py --lang en   # OpenAI JSONL -> mlx-lm format
python src/preprocessing/format_mlx_data.py --lang es
python src/preprocessing/prepare_ablation_data.py       # legacy EN: SFT-pure + Mars-PO team
```

**Outputs** (per language):

| File | Purpose |
|------|---------|
| `data/preference/{lang}/sft_<cluster>_{train,val}.jsonl` | SFT per cluster (with cluster-mixing) |
| `data/preference/{lang}/sft_balanced_<cluster>_{train,val}.jsonl` | Balanced-SFT ablation |
| `data/preference/{lang}/dpo_<cluster>_{train,val}.jsonl` | Individual DPO pairs |
| `data/splits/{lang}/train_set.json`, `test_set.json` | Group-level splits (EN 2,576/342; ES 3,021/297) |
| `data/splits/{lang}/split_info.json` | Group assignments (EN 46/6/6, ES 53/6/6) |
| `data/splits/{lang}/cluster_meta.json` | Cluster sizes + YES-rate targets |
| `data/mlx/{lang}/` | mlx-lm-formatted datasets |

The legacy flat files (`data/preference/sft_<legacy-name>_*.jsonl`,
`data/splits/{train,test}_set.json`) belong to the frozen GPT-EN runs.

## Step 3: Phase 1 — SFT Fine-Tuning

Local (per language, handled by `run_local_experiments.py`): `mlx_sft.py`
trains one LoRA adapter per cluster into `artifacts/adapters/{lang}/`.

OpenAI (legacy EN flow):

```bash
python src/training/sft_finetune.py                 # launch all 3 agents
python src/training/sft_finetune.py --status        # check progress
python src/training/sft_finetune.py --wait          # launch + block
```

**Output:** `artifacts/model_ids/sft_model_ids.json` (gpt-4o-mini base)

To run the same step on gpt-4.1-mini, edit `src/config.py`:
```python
BASE_MODEL = "gpt-4.1-mini-2025-04-14"
```
then change the `suffix=` line in `src/training/sft_finetune.py`
(e.g. `behav-po-sft41-{c_lower}`) and rerun. That writes
`artifacts/model_ids/sft_41mini_model_ids.json`.

## Step 4: Phase 2 — DPO / Mars-PO

**DPO is gpt-4.1-mini only.** `gpt-4o-mini` has no DPO endpoint.

```bash
python src/training/dpo_finetune.py
python src/training/dpo_finetune.py --status
```

Uses `src/config.py:DPO_BETA` (default 0.1) and
`data/preference/dpo_<cluster>_{train,val}.jsonl`.

**Mars-PO (correct)** — point the script at the concatenated files:
`data/preference/dpo_marspo_<cluster>_{train,val}.jsonl` and change the suffix
to `behav-po-marspo-{c_lower}`.

**β-sweep (0.3, 0.5)** — set `config.DPO_BETA` and change the suffix to
`bpo-dpo-b{03,05}-{c_lower}` between runs.

## Step 5: Phase 3 — GRPO via Rejection Sampling

GRPO is approximated by rejection sampling with the **convex reward**

```
R_j = (1 - α) · r_indiv_j + α · r_team        (α = config.GRPO_ALPHA = 0.2)
```

where `r_indiv_j = 1` if agent j matches its cluster's majority vote and
`r_team = 1` if the 3-agent majority matches the human majority. Samples with
`R_j >= 0.5` are kept and the agents are re-fine-tuned on them. The legacy
GPT-EN runs used the unnormalized `R = r_indiv + α'·r_team`; at α'=0.2
(convex-equivalent α=0.17) both forms keep the identical sample set.

Local: the `grpo_sample_*` / `grpo_*` steps of `run_local_experiments.py`
(`mlx_grpo.py`). OpenAI (legacy EN flow):

```bash
python src/training/grpo_rejection_sampling.py --iteration 1 --sample-size 500
python src/training/grpo_rejection_sampling.py --status
```

The sampling pass writes `data/preference/phase3_raw_samples.json` (cached —
reused on subsequent α values).

**α-sweep** on gpt-4.1-mini — once per-α data files exist at
`data/preference/grpo_41mini_alpha{00,02,05,10}_<cluster>.jsonl`, let the
orchestrator submit the fine-tune jobs (Step 8).

## Step 5b: OpenAI Spanish Suite (GPT-ES)

The full Spanish `gpt-4.1-mini` pipeline (SFT -> DPO -> GRPO α=0.2 -> predictions
and results), plus the EN per-text prediction backfill, is one resume-safe
script:

```bash
python3 src/training/openai_es_suite.py            # run all pending steps
python3 src/training/openai_es_suite.py --only sft_wait
python3 src/training/openai_es_suite.py --status
```

State lives in `artifacts/openai_es_state.json`; model IDs land in
`artifacts/model_ids/es_{sft,dpo,grpo_a20}_41mini_model_ids.json` and results
in `artifacts/results/gpt41mini_{sft,dpo,grpo_a20}_es_results.json`.

## Step 6: Evaluation

Local (both languages, run by `run_local_experiments.py`):

```bash
.venv-mlx/bin/python src/evaluation/evaluate_local.py --lang en --stage sft
python3 src/evaluation/behavioral_fidelity.py --lang en    # agents vs annotator distributions
python3 src/evaluation/behavioral_fidelity.py --lang es
```

OpenAI (legacy EN flow):

```bash
python src/evaluation/evaluate_agents.py --model sft     # after SFT
python src/evaluation/evaluate_agents.py --model dpo     # after DPO
python src/evaluation/evaluate_agents.py --model grpo    # after GRPO
python src/evaluation/evaluate_agents.py --baselines     # Always-NO / zero-shot / persona
python src/evaluation/evaluate_agents.py --all           # everything
```

**Outputs:** `artifacts/results/<method>_results.json`,
`artifacts/predictions/<method>_predictions.json`

### Key Metrics

| Metric | What it measures | Target |
|--------|-----------------|--------|
| MVA | Team majority vote vs human majority | > 85% |
| CAA | Agent prediction vs its cluster's vote | > 75% per cluster |
| YES-Rate Calibration | Agent YES rate vs cluster YES rate | < 5% error |
| Disagreement Rate | How often agents disagree | 15–40% |
| F1-macro | Team prediction F1 | > 0.80 |

### Statistical analysis

```bash
python src/evaluation/analysis.py
```
Writes `artifacts/results/confidence_intervals.json` and
`artifacts/results/significance_tests.json`.

### Plots

```bash
python src/evaluation/generate_plots.py
```
Writes `paper/figures/{mva_comparison,caa_comparison,yes_rates,disagreement,alpha_sweep,beta_sweep}.pdf`.

## Step 7: Optional — Deliberation

```bash
python src/training/deliberation.py --generate --source sft
python src/training/deliberation.py --evaluate --source sft
```

## Step 8: Orchestrator (end-to-end α/β sweeps)

Idempotent driver for the Phase 3 α-sweep and Phase 4 β-sweep. Safe to
restart after reboot — reads `artifacts/model_ids/phase{3,4}_*_jobs.json`
instead of resubmitting.

```bash
python src/pipeline/orchestrator.py 2>&1 | tee -a orchestrator.log
```

Stages:
1. Wait for Phase 3 sampling.
2. Submit Phase 3 batch 1 (α = 0.0, 0.5).
3. Wait for Phase 4 + batch 1 to complete.
4. Evaluate both.
5. Submit Phase 3 batch 2 (α = 1.0).
6. Wait + evaluate.

**Manual restart fallback:**
```bash
python src/pipeline/resume_pipeline.py
```
Scans OpenAI for completed jobs, saves missing `*_model_ids.json`, runs
`evaluate_agents.py` on anything without a results file, submits the next
batch if slots are free.

## Step 9: Verification

Run after any pipeline stage to catch regressions:

```bash
python src/pipeline/verify_pipeline.py
```

Checks clustering outputs, data shapes, splits, and result JSONs for all
four settings. Data checks are skipped with a warning until `data/` has been
regenerated from the EXIST source (see README).

## Naming Conventions

| Token | Meaning |
|---|---|
| `sft_` (no base tag) | SFT on gpt-4o-mini (legacy EN) |
| `base_41mini_` | Zero-shot ensemble on gpt-4.1-mini (GPT-EN baseline) |
| `persona_41mini_` | Persona-prompt baseline on gpt-4.1-mini (GPT-EN) |
| `sft_41mini_` | SFT on gpt-4.1-mini (GPT-EN) |
| `sft_pure_41mini_` | SFT-pure ablation (no cluster mixing) |
| `dpo_41mini_` | DPO individual-only on gpt-4.1-mini |
| `marspo_41mini_` | Mars-PO correct (individual + team DPO) |
| `dpo_bb03_41mini_` / `dpo_bb05_41mini_` | DPO β-sweep |
| `grpo_` | GRPO α'=0.2 on gpt-4o-mini (legacy EN) |
| `grpo_41mini_alpha{00,02,05,10}_` | GRPO α'-sweep on gpt-4.1-mini (convex α = 0, 0.17, 0.33, 0.5) |
| `gpt41mini_{sft,dpo,grpo_a20}_es_` | GPT-ES suite (`openai_es_suite.py`) |
| `gpt41mini_base_es_` | Zero-shot ensemble on gpt-4.1-mini (GPT-ES baseline) |
| `local_qwen3_{base,sft,dpo,grpo_a20}_{en,es}_` | Local Qwen3-8B runs |
| `local_qwen3_sft_balanced_en_` | Balanced-SFT ablation (Qwen-EN) |
| `behavioral_fidelity_{en,es}` | Agents vs annotator feature distributions |
| `clustering_ablation_{en,es}` | 2-feature clustering ablation (report-only) |

Cluster keys are `cluster1..3` (ascending YES rate) everywhere; legacy names
in the frozen EN artifacts are normalized at load via `src/config.py`.
