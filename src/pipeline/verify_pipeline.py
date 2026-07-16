#!/usr/bin/env python3
"""
Pipeline Integrity Checker
===========================
Verifies the BehAv-PO pipeline is in a consistent state. Run after any
pipeline stage to catch regressions.

Data checks are skipped with a warning until data/ has been regenerated
from the EXIST 2024 source (see README); the shipped result files under
artifacts/results/ are always checked.

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
"""

import json
import os
import sys

# Make `import config` and `from models.agent ...` work from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def check(name, condition, detail=""):
    status = OK if condition else FAIL
    print(f"  {status} {name}" + (f" - {detail}" if detail else ""))
    return condition


def warn(name, detail=""):
    print(f"  {WARN} {name}" + (f" - {detail}" if detail else ""))


def main():
    print("=" * 60)
    print("BehAv-PO Pipeline Verification")
    print("=" * 60)

    all_passed = True
    have_data = os.path.exists("data/raw/EXIST2024_training.json")
    if not have_data:
        warn("data/raw/EXIST2024_training.json missing",
             "data checks skipped; request EXIST 2024 from the organizers")

    # 1. Per-language clustering outputs (neutral cluster keys)
    print("\n1. Per-language clustering outputs")
    import csv
    expected_rows = {"en": 348, "es": 390}
    for lang, n_expected in expected_rows.items():
        path = f"data/behavioral_features_{lang}.csv"
        if not os.path.exists(path):
            warn(path, "not generated yet")
            continue
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
            clusters = {r["cluster"] for r in rows}
            all_passed &= check(
                f"  {lang}: cluster values are cluster1..3",
                clusters == {"cluster1", "cluster2", "cluster3"},
                f"got {sorted(clusters)}",
            )
            all_passed &= check(
                f"  {lang}: {n_expected} annotators",
                len(rows) == n_expected,
                f"got {len(rows)}",
            )
        except Exception as e:
            all_passed &= check(f"  {lang}: CSV loadable", False, str(e))

    # 2. Per-language training data
    print("\n2. Per-language training data")
    for lang in ["en", "es"]:
        for cluster in ["cluster1", "cluster2", "cluster3"]:
            path = f"data/preference/{lang}/sft_{cluster}_train.jsonl"
            if not os.path.exists(path):
                warn(path, "not generated yet")
                continue
            with open(path) as f:
                lines = sum(1 for _ in f)
            all_passed &= check(f"{path}: >500 examples", lines > 500, f"{lines}")

    # 3. Per-language data splits
    print("\n3. Per-language data splits")
    for lang in ["en", "es"]:
        for name in ["test_set", "train_set", "cluster_meta"]:
            f = f"data/splits/{lang}/{name}.json"
            if os.path.exists(f):
                all_passed &= check(f, True)
            else:
                warn(f, "not generated yet")

    # 4. Evaluation results (shipped; always checked)
    print("\n4. Evaluation results")
    result_files = [
        "artifacts/results/baseline_results.json",
        "artifacts/results/sft_results.json",
        "artifacts/results/grpo_results.json",
        "artifacts/results/sft_41mini_results.json",
        "artifacts/results/sft_pure_41mini_results.json",
        "artifacts/results/dpo_41mini_results.json",
        "artifacts/results/dpo_bb03_41mini_results.json",
        "artifacts/results/dpo_bb05_41mini_results.json",
        "artifacts/results/marspo_41mini_results.json",
        "artifacts/results/grpo_41mini_alpha00_results.json",
        "artifacts/results/grpo_41mini_alpha02_results.json",
        "artifacts/results/grpo_41mini_alpha05_results.json",
        "artifacts/results/grpo_41mini_alpha10_results.json",
        "artifacts/results/confidence_intervals.json",
        "artifacts/results/significance_tests.json",
    ]
    for f in result_files:
        exists = os.path.exists(f)
        all_passed &= check(f, exists)
        if exists:
            try:
                with open(f) as fh:
                    json.load(fh)
            except Exception as e:
                all_passed &= check(f"  {os.path.basename(f)}: well-formed JSON", False, str(e))

    # 4b. Every fine-tuned result reports the core metrics
    print("\n4b. Result metric coverage")
    required_keys = ["mva", "f1_macro", "disagreement_rate"]
    fine_tuned = [
        "artifacts/results/sft_41mini_results.json",
        "artifacts/results/sft_pure_41mini_results.json",
        "artifacts/results/dpo_41mini_results.json",
        "artifacts/results/dpo_bb03_41mini_results.json",
        "artifacts/results/dpo_bb05_41mini_results.json",
        "artifacts/results/marspo_41mini_results.json",
        "artifacts/results/grpo_41mini_alpha00_results.json",
        "artifacts/results/grpo_41mini_alpha02_results.json",
        "artifacts/results/grpo_41mini_alpha05_results.json",
        "artifacts/results/grpo_41mini_alpha10_results.json",
    ]
    for f in fine_tuned:
        if not os.path.exists(f):
            continue
        with open(f) as fh:
            d = json.load(fh)
        missing = [k for k in required_keys if k not in d]
        all_passed &= check(
            f"  {os.path.basename(f)}: mva/f1_macro/disagreement_rate",
            not missing,
            f"missing {missing}" if missing else "",
        )

    # 4c. Bilingual / local experiment results
    print("\n4c. Bilingual + local results")
    local_gpt_results = []
    for lang in ["en", "es"]:
        for stage in ["base", "sft", "dpo", "grpo_a20"]:
            local_gpt_results.append(f"artifacts/results/local_qwen3_{stage}_{lang}_results.json")
    local_gpt_results.append("artifacts/results/local_qwen3_sft_balanced_en_results.json")
    for stage in ["sft", "dpo", "grpo_a20"]:
        local_gpt_results.append(f"artifacts/results/gpt41mini_{stage}_es_results.json")
    analysis_results = [
        "artifacts/results/behavioral_fidelity_en.json",
        "artifacts/results/behavioral_fidelity_es.json",
        "artifacts/results/clustering_ablation_en.json",
        "artifacts/results/clustering_ablation_es.json",
    ]
    required_metrics = ["mva", "f1_macro", "balanced_accuracy", "disagreement_rate"]
    for f in local_gpt_results:
        exists = os.path.exists(f)
        all_passed &= check(f, exists)
        if not exists:
            continue
        try:
            with open(f) as fh:
                d = json.load(fh)
        except Exception as e:
            all_passed &= check(f"  {os.path.basename(f)}: well-formed JSON", False, str(e))
            continue
        missing = [k for k in required_metrics if k not in d]
        all_passed &= check(
            f"  {os.path.basename(f)}: mva/f1_macro/balanced_accuracy/disagreement_rate",
            not missing,
            f"missing {missing}" if missing else "",
        )
    for f in analysis_results:
        exists = os.path.exists(f)
        all_passed &= check(f, exists)
        if exists:
            try:
                with open(f) as fh:
                    json.load(fh)
            except Exception as e:
                all_passed &= check(f"  {os.path.basename(f)}: well-formed JSON", False, str(e))

    # 5. Synthetic-data gate
    print("\n5. Synthetic-data gate")
    tainted = []
    for name in sorted(os.listdir("artifacts/results")):
        path = os.path.join("artifacts/results", name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                content = fh.read()
        except Exception:
            continue
        if '"synthetic": true' in content:
            tainted.append(name)
    all_passed &= check(
        "no synthetic results in artifacts/results/",
        not tainted,
        f"tainted: {tainted}" if tainted else "",
    )

    # 6. Python modules
    print("\n6. Python modules")
    try:
        import config  # noqa
        all_passed &= check("config.py imports", True)
    except Exception as e:
        all_passed &= check("config.py imports", False, str(e))
    try:
        from models.agent import Agent  # noqa
        all_passed &= check("models.agent imports", True)
        from evaluation.metrics import compute_all_metrics  # noqa
        all_passed &= check("evaluation.metrics imports", True)
    except ModuleNotFoundError as e:
        warn("module imports", f"{e}; run: pip install -r requirements.txt")
    except Exception as e:
        all_passed &= check("module imports", False, str(e))

    # Summary
    print("\n" + "=" * 60)
    if all_passed:
        print(f"{OK} ALL CHECKS PASSED")
        print("=" * 60)
        sys.exit(0)
    else:
        print(f"{FAIL} ONE OR MORE CHECKS FAILED")
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
