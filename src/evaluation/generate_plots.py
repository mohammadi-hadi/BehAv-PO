#!/usr/bin/env python3
"""
Thin wrapper that regenerates every paper figure by delegating to the canonical
plot scripts in report_assets/figures/ (the single source of truth for style
and method ordering). Output PDFs/PNGs land directly in paper/figures/.

Run from repo root:
    python3 src/evaluation/generate_plots.py
"""
import importlib.util
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REPORT_ASSETS = os.path.join(REPO_ROOT, "report_assets")
FIG_SRC = os.path.join(REPORT_ASSETS, "figures")
PAPER_FIG = os.path.join(REPO_ROOT, "paper", "figures")

# Canonical plot list (kept in sync with report_assets/generate_all.py).
PLOTS = [
    "plot_team_f1",
    "plot_cluster_f1",
    "plot_cluster_vs_team",
    "plot_balance_scatter",
    "plot_cluster_f1_breakdown",
    "plot_cluster_coverage",
    "plot_alpha_sweep",
    "plot_beta_sweep",
    "plot_yes_rates",
    "plot_agreement",
]


def _run(module_name):
    # Make report_assets/ importable so plot scripts find utils.py.
    if REPORT_ASSETS not in sys.path:
        sys.path.insert(0, REPORT_ASSETS)
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(FIG_SRC, f"{module_name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(PAPER_FIG)


def main():
    os.makedirs(PAPER_FIG, exist_ok=True)
    for name in PLOTS:
        print(f"\n--> {name}")
        _run(name)
    print(f"\nAll plots saved to {PAPER_FIG}/")


if __name__ == "__main__":
    main()
