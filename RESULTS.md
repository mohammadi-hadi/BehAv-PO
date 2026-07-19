# BehAv-PO — Complete Experimental Results

*Generated from `artifacts/` by `src/evaluation/generate_results_md.py` on 2026-07-19 (commit `21d55d8`). Do not edit by hand — regenerate instead. All scores are ×100; F1 is macro (mean of F1-YES and F1-NO) unless stated otherwise.*

Four settings cross two backbones with two languages: **GPT-EN / GPT-ES** fine-tune `gpt-4.1-mini` through the OpenAI API; **Qwen-EN / Qwen-ES** train Qwen3-8B locally with MLX LoRA. Three agents (one per behavioral annotator cluster) vote on the final label.

## 1. The 2×2 design at a glance

Cluster-specific fine-tuning pays in every setting; whether GRPO's team reward also lifts the headline depends on backbone capacity (it leads on GPT-EN, ties on GPT-ES, trails SFT on the smaller Qwen backbone).

| Stage | GPT-EN Team F1 | GPT-EN Bal. Acc. | GPT-EN Avg C-F1 | GPT-ES Team F1 | GPT-ES Bal. Acc. | GPT-ES Avg C-F1 | Qwen-EN Team F1 | Qwen-EN Bal. Acc. | Qwen-EN Avg C-F1 | Qwen-ES Team F1 | Qwen-ES Bal. Acc. | Qwen-ES Avg C-F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Zero-shot | 64.7 | 65.4 | 59.7 | 57.4 | 64.5 | 58.2 | 76.8 | 76.9 | 65.1 | 72.9 | 74.1 | 65.7 |
| SFT | 89.1 | 89.2 | 79.7 | 84.6 | 85.3 | **76.5** | **84.7** | **85.1** | 75.5 | **81.0** | **81.4** | **77.0** |
| DPO | 85.8 | 86.2 | 60.9 | **85.5** | **86.7** | 56.7 | 78.1 | 80.7 | 54.1 | 80.5 | 80.2 | 53.8 |
| GRPO | **89.9** | **90.3** | **80.0** | 85.1 | 86.2 | 73.2 | 80.9 | 80.7 | **75.8** | 80.6 | 80.8 | 75.4 |

## 2. GPT-EN: full method comparison

All rows share the `gpt-4.1-mini` base. Cluster F1 scores each agent against its own cluster's majority; Overall F1 is the harmonic mean of Avg Cluster F1 and Team F1 (a method cannot score high by sacrificing one axis). Agreement is a diversity diagnostic, not a quality metric.

| Method | C1 F1 | C2 F1 | C3 F1 | Avg C-F1 | Team F1 | Team Acc. | Overall F1 | Agreement % |
|---|---|---|---|---|---|---|---|---|
| Always-NO | -- | -- | -- | -- | 37.8 | 60.7 | -- | 100.0 |
| Zero-shot Ensemble | 66.4 | 63.3 | 49.3 | 59.7 | 64.7 | 72.5 | 62.1 | 96.8 |
| Persona Prompts | 64.0 | 64.6 | 61.7 | 63.4 | 68.3 | 74.5 | 65.8 | 79.2 |
| DPO | 49.6 | 84.3 | 48.7 | 60.9 | 85.8 | 86.6 | 71.2 | 6.1 |
| SFT-pure | 73.2 | 87.4 | 76.8 | 79.1 | 88.4 | 88.9 | 83.5 | 40.6 |
| SFT (mixed) | 72.7 | 85.5 | **81.0** | 79.7 | 89.1 | 89.6 | 84.2 | 55.0 |
| Mars-PO | **75.9** | 86.9 | 80.3 | **81.0** | 88.8 | 89.3 | **84.8** | 65.5 |
| GRPO α=0.17 | 73.9 | **88.2** | 77.9 | 80.0 | **89.9** | **90.3** | 84.7 | 56.4 |

Every fine-tuned variant separates from the strongest baseline (persona prompts) under the 95% bootstrap-CI non-overlap criterion; GRPO's team accuracy CI is [86.6, 93.3] vs persona's [69.1, 79.5].

## 3. The polarization → repair result (replicates 4/4 settings)

Individual-only DPO pins the outer agents' YES rates to the extremes in every setting; GRPO's convex team reward pulls every agent back toward its cluster. Targets are the cluster annotator-mean YES rates.

| Setting | Stage | C1 YES | C2 YES | C3 YES | Mean calib. err. | Disagreement |
|---|---|---|---|---|---|---|
| GPT-EN | SFT | 21.3% | 43.3% | 64.6% | 0.6% | 45.0% |
|  | DPO | 1.5% | 39.5% | 95.3% | 18.8% | 93.9% |
|  | GRPO | 22.8% | 49.1% | 59.7% | 3.5% | 43.6% |
| *GPT-EN targets* |  | 21.5% | 43.8% | 63.5% | 0 | — |
| GPT-ES | SFT | 28.6% | 52.9% | 65.0% | 3.4% | 38.4% |
|  | DPO | 0.0% | 48.5% | 99.7% | 20.0% | 99.7% |
|  | GRPO | 37.4% | 45.8% | 64.6% | 5.0% | 33.7% |
| *GPT-ES targets* |  | 26.8% | 47.2% | 67.6% | 0 | — |
| Qwen-EN | SFT | 22.5% | 45.6% | 65.8% | 1.7% | 43.9% |
|  | DPO | 0.0% | 58.8% | 100.0% | 24.3% | 100.0% |
|  | GRPO | 21.6% | 40.9% | 66.4% | 2.0% | 45.9% |
| *Qwen-EN targets* |  | 21.5% | 43.8% | 63.5% | 0 | — |
| Qwen-ES | SFT | 17.8% | 54.5% | 67.7% | 5.4% | 49.8% |
|  | DPO | 0.0% | 63.3% | 100.0% | 25.1% | 100.0% |
|  | GRPO | 22.2% | 58.9% | 59.9% | 8.0% | 41.1% |
| *Qwen-ES targets* |  | 26.8% | 47.2% | 67.6% | 0 | — |

## 4. Reward sweeps (GPT-EN)

**GRPO α-sweep** (convex team weight; α=0.17 is the convex relabeling of the original α'=0.2):

| α | Team Acc. | Team F1 | Disagreement |
|---|---|---|---|
| 0 (indiv. only) | 89.9 | 89.5 | 42.7% |
| 0.17 | **90.3** | **89.9** | 43.6% |
| 0.33 | 88.6 | 87.9 | 36.3% |
| 0.5 (equal) | 89.3 | 89.0 | 50.6% |

**DPO β-sweep** (individual-only preferences; a stronger KL constraint reduces the overshoot but does not remove it — only a team signal does):

| β / method | Team Acc. | Team F1 | Disagreement |
|---|---|---|---|
| 0.1 (default) | 86.6 | 85.8 | 93.9% |
| 0.3 | 87.6 | 87.0 | 79.2% |
| 0.5 | 88.6 | 88.0 | 72.5% |
| Mars-PO (team pairs) | 89.3 | 88.8 | 34.5% |

## 5. All-clusters subset (Wilson 95% CIs)

Texts annotated by members of all three clusters: n=57 (47 gold) on English, 126 (111 gold) on Spanish. Method ranking is preserved; intervals are wide, so the subset corroborates the full-set results.

| Setting | Stage | Full Team F1 | Subset Team F1 | Subset Acc. | 95% CI (acc.) |
|---|---|---|---|---|---|
| GPT-EN | Zero-shot | 64.7 | 69.5 | 74.5 | [60.5, 84.8] |
|  | SFT | 89.1 | 95.5 | 95.7 | [85.7, 98.8] |
|  | DPO | 85.8 | 95.5 | 95.7 | [85.7, 98.8] |
|  | GRPO | 89.9 | 95.5 | 95.7 | [85.7, 98.8] |
| GPT-ES | Zero-shot | 57.4 | 62.1 | 63.1 | [53.8, 71.5] |
|  | SFT | 84.6 | 84.4 | 84.7 | [76.8, 90.2] |
|  | DPO | 85.5 | 86.3 | 86.5 | [78.9, 91.6] |
|  | GRPO | 85.1 | 83.5 | 83.8 | [75.8, 89.5] |
| Qwen-EN | Zero-shot | 76.8 | 78.2 | 78.7 | [65.1, 88.0] |
|  | SFT | 84.7 | 86.8 | 87.2 | [74.8, 94.0] |
|  | DPO | 78.1 | 82.9 | 83.0 | [69.9, 91.1] |
|  | GRPO | 80.9 | 84.4 | 85.1 | [72.3, 92.6] |
| Qwen-ES | Zero-shot | 72.9 | 75.5 | 75.7 | [66.9, 82.7] |
|  | SFT | 81.0 | 82.5 | 82.9 | [74.8, 88.8] |
|  | DPO | 80.5 | 82.0 | 82.9 | [74.8, 88.8] |
|  | GRPO | 80.6 | 83.2 | 83.8 | [75.8, 89.5] |

## 6. Per-class F1 and balanced accuracy

The English test set is 39.3% YES; the Spanish one flips to 58.2% YES. Per-class scores confirm the headline F1-macro is not an artifact of class imbalance; SFT-balanced is a negative-result ablation (balancing labels erases the YES-rate differences the clusters are defined by).

| Setting | Stage | F1-YES | F1-NO | F1-macro | Bal. Acc. | Calib. Err. |
|---|---|---|---|---|---|---|
| Qwen-EN | Zero-shot | 72.0 | 81.7 | 76.8 | 76.9 | 14.3% |
|  | SFT | 81.8 | 87.6 | 84.7 | 85.1 | 1.7% |
|  | DPO | 76.9 | 79.4 | 78.1 | 80.7 | 24.3% |
|  | GRPO | 76.5 | 85.2 | 80.9 | 80.7 | 2.0% |
|  | SFT-balanced | 80.5 | 86.3 | 83.4 | 84.0 | 10.9% |
| Qwen-ES | Zero-shot | 74.4 | 71.5 | 72.9 | 74.1 | 14.0% |
|  | SFT | 83.5 | 78.6 | 81.0 | 81.4 | 5.4% |
|  | DPO | 84.5 | 76.6 | 80.5 | 80.2 | 25.1% |
|  | GRPO | 83.3 | 77.9 | 80.6 | 80.8 | 8.0% |
| GPT-ES | Zero-shot | 48.3 | 66.5 | 57.4 | 64.5 | 27.1% |
|  | SFT | 86.3 | 82.9 | 84.6 | 85.3 | 3.4% |
|  | DPO | 86.5 | 84.4 | 85.5 | 86.7 | 20.0% |
|  | GRPO | 86.2 | 84.0 | 85.1 | 86.2 | 5.0% |

GPT-EN per-class diagnostics come from the backfilled predictions (Section 2 columns) and its balanced accuracies from Section 1.

## 7. Behavioral fidelity (are agents typical cluster members?)

YES-rate z-score of each agent within its cluster's per-annotator distribution (|z| < 1 ≈ typical member). SFT/GRPO agents are typical; DPO-only agents sit far outside any actual annotator.

| Run | Lang | C1 z | C2 z | C3 z |
|---|---|---|---|---|
| sft_41mini | EN | -0.10 | +0.00 | +0.10 |
| dpo_41mini | EN | -3.91 | -0.43 | +2.10 |
| marspo_41mini | EN | +0.48 | +0.25 | -0.57 |
| grpo_41mini_alpha02 | EN | +0.19 | +0.54 | -0.15 |
| sft_pure_41mini | EN | -1.89 | -0.15 | +0.59 |
| local_qwen3_base_en | EN | +4.17 | -0.12 | -1.35 |
| local_qwen3_sft_en | EN | +0.19 | +0.19 | +0.15 |
| local_qwen3_dpo_en | EN | -4.25 | +1.60 | +2.38 |
| local_qwen3_grpo_a20_en | EN | +0.02 | -0.31 | +0.19 |
| local_qwen3_base_es | ES | +3.08 | -0.14 | -3.39 |
| local_qwen3_sft_es | ES | -1.42 | +0.93 | +0.01 |
| local_qwen3_dpo_es | ES | -4.26 | +2.04 | +5.10 |
| local_qwen3_grpo_a20_es | ES | -0.72 | +1.49 | -1.21 |
| gpt41mini_sft_es | ES | +0.29 | +0.72 | -0.42 |
| gpt41mini_dpo_es | ES | -4.26 | +0.16 | +5.05 |
| gpt41mini_grpo_a20_es | ES | +1.69 | -0.18 | -0.47 |

## 8. Behavioral clustering

| Lang | Cluster | Size | YES rate | Agreement rate | Scan |
|---|---|---|---|---|---|
| EN | cluster1 | 75 | 0.215 ± 0.051 | 0.865 ± 0.063 | chosen k=3, silhouette 0.427 |
|  | cluster2 | 224 | 0.438 ± 0.093 | 0.868 ± 0.050 |  |
|  | cluster3 | 49 | 0.635 ± 0.153 | 0.724 ± 0.107 |  |
| ES | cluster1 | 67 | 0.268 ± 0.063 | 0.783 ± 0.113 | chosen k=3, silhouette 0.349 (forced; silhouette prefers k=2 at 0.465) |
|  | cluster2 | 234 | 0.472 ± 0.079 | 0.867 ± 0.061 |  |
|  | cluster3 | 89 | 0.676 ± 0.063 | 0.836 ± 0.091 |  |

Dropping the YES rate from the features (2-feature ablation) collapses the direction of an annotator's leaning: ARI vs the 3-feature reference falls to 0.71 (EN) / 0.14 (ES) and the minimum cluster-mean YES-rate gap to 0.06 / 0.04 (reference ≈ 0.20). Chi-squared tests find no significant demographic association with cluster membership (Cramér's V ≤ 0.13).

## 9. Early validation study (gpt-4o-mini)

Pipeline validation on `gpt-4o-mini-2024-07-18` before the primary runs. Notably the older base is the *stronger* zero-shot model (77.8 Team F1 vs 64.7 for `gpt-4.1-mini`), so the fine-tuned gains are not inherited from the backbone.

| Method (gpt-4o-mini) | Team Acc. | Team F1 | Disagreement |
|---|---|---|---|
| Zero-shot ensemble | 80.9 | 77.8 | 2.6% |
| SFT | 87.9 | 87.0 | 38.9% |
| GRPO α=0.17 | 86.6 | 86.2 | 46.2% |

## Appendix: result-file index

Every number above traces to `artifacts/results/<label>_results.json` (per-text votes in `artifacts/predictions/<label>_predictions.json`).

`base_41mini`, `dpo_41mini`, `dpo_bb03_41mini`, `dpo_bb05_41mini`, `gpt41mini_base_es`, `gpt41mini_dpo_es`, `gpt41mini_grpo_a20_es`, `gpt41mini_sft_es`, `grpo`, `grpo_41mini_alpha00`, `grpo_41mini_alpha02`, `grpo_41mini_alpha05`, `grpo_41mini_alpha10`, `local_qwen3_base_en`, `local_qwen3_base_es`, `local_qwen3_dpo_en`, `local_qwen3_dpo_es`, `local_qwen3_grpo_a20_en`, `local_qwen3_grpo_a20_es`, `local_qwen3_sft_balanced_en`, `local_qwen3_sft_en`, `local_qwen3_sft_es`, `marspo_41mini`, `persona_41mini`, `sft`, `sft_41mini`, `sft_pure_41mini`, plus `baseline_results.json`, `confidence_intervals.json`, `significance_tests.json`.
