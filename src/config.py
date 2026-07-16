"""
BehAv-PO Configuration
======================
Shared hyperparameters, paths, and model settings for the
multi-agent behavioral preference optimization pipeline.

All paths are absolute (resolved relative to the repo root) so scripts
work from any current working directory.
"""

import os
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR                 = REPO_ROOT / "data"
RAW_DATA_DIR             = DATA_DIR / "raw"
SPLITS_DIR               = DATA_DIR / "splits"
PREFERENCE_DIR           = DATA_DIR / "preference"

DATA_PATH                = RAW_DATA_DIR / "EXIST2024_training.json"
BEHAVIORAL_FEATURES_PATH = DATA_DIR / "behavioral_features.csv"

ARTIFACTS_DIR            = REPO_ROOT / "artifacts"
MODEL_IDS_DIR            = ARTIFACTS_DIR / "model_ids"
RESULTS_DIR              = ARTIFACTS_DIR / "results"
SYNTHETIC_RESULTS_DIR    = ARTIFACTS_DIR / "results_synthetic"
PREDICTIONS_DIR          = ARTIFACTS_DIR / "predictions"
ADAPTERS_DIR             = ARTIFACTS_DIR / "adapters"
LOGS_DIR                 = ARTIFACTS_DIR / "logs"
MLX_DATA_DIR             = DATA_DIR / "mlx"

PAPER_DIR                = REPO_ROOT / "paper"
PAPER_FIGURES_DIR        = PAPER_DIR / "figures"

# Back-compat: some scripts use string-valued dir names. Prefer the Path
# attributes above in new code.
TRAINING_DIR             = str(REPO_ROOT / "src" / "training")
EVALUATION_DIR           = str(REPO_ROOT / "src" / "evaluation")

# ── OpenAI ────────────────────────────────────────────────────
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
BASE_MODEL = "gpt-4o-mini-2024-07-18"

SYSTEM_PROMPT = (
    "You are a content annotator. "
    "Classify whether the given tweet contains sexism. "
    "Respond with exactly YES or NO."
)

# ── Local model (MLX, Apple Silicon) ─────────────────────────
LOCAL_BASE_MODEL = "Qwen/Qwen3-8B"
LOCAL_MODEL_LABEL = "qwen3"          # used in result-file names
MLX_PYTHON = str(REPO_ROOT / ".venv-mlx" / "bin" / "python")

# ── Dataset ───────────────────────────────────────────────────
LANGUAGE = "en"           # legacy default (OpenAI EN experiments)
LANGUAGES = ["en", "es"]
LANG_SPLIT_TAG = {"en": "EN", "es": "ES"}
EXPECTED_ANNOTATORS = {"en": 348, "es": 390}
MIN_LABELS = 10
RANDOM_STATE = 42

# ── Clusters (neutral naming; ordered by ascending YES rate) ──
N_CLUSTERS = 3
CLUSTER_IDS = [1, 2, 3]
CLUSTER_KEYS = [f"cluster{i}" for i in CLUSTER_IDS]     # file/JSON-safe keys
# Display names: change here (and only here) to rename clusters everywhere.
CLUSTER_DISPLAY = {f"cluster{i}": f"Cluster {i}" for i in CLUSTER_IDS}
CLUSTER_SHORT = {f"cluster{i}": f"C{i}" for i in CLUSTER_IDS}

# Legacy names used by the frozen OpenAI EN artifacts (results, splits,
# model-id JSONs). Kept only so old files can be normalized at load time.
LEGACY_CLUSTER_TO_KEY = {
    "Conservative": "cluster1",
    "Mainstream": "cluster2",
    "Sensitive": "cluster3",
}
KEY_TO_LEGACY = {v: k for k, v in LEGACY_CLUSTER_TO_KEY.items()}

# Deprecated alias — legacy OpenAI scripts import this. Do not use in new code.
CLUSTER_NAMES = list(LEGACY_CLUSTER_TO_KEY.keys())

# Force the number of behavioral clusters per language (None = pick k by max
# silhouette). ES is forced to k=3 for architecture parity (3 agents +
# majority vote across languages) even though its silhouette-best k is 2;
# the honest scan stays recorded in artifacts/clustering/es/kmeans_scan.json.
FORCE_K = {"en": None, "es": 3}

# Train/Val/Test split — by annotator group index
# EN: 58 groups (46/6/6). ES: 65 groups (53/6/6).
N_TRAIN_GROUPS = 46
N_VAL_GROUPS = 6
N_TEST_GROUPS = 6
N_GROUP_SPLIT = {"en": (46, 6, 6), "es": (53, 6, 6)}

# ── SFT Hyperparameters ──────────────────────────────────────
SFT_EPOCHS = 3
SFT_SHARED_POSITIVE_RATIO = 0.30
SFT_TEAM_MAJORITY_RATIO = 0.20

# ── DPO Hyperparameters ──────────────────────────────────────
DPO_BETA = 0.1

# ── GRPO / Rejection Sampling ────────────────────────────────
# Reward: R = (1 - alpha) * r_indiv + alpha * r_team  (convex combination).
# Legacy OpenAI runs used R = r_indiv + alpha' * r_team; at alpha'=0.2 with
# threshold 0.5 both formulas keep the identical sample set, so those results
# remain valid (alpha = alpha' / (1 + alpha')).
GRPO_K_SAMPLES = 8
GRPO_ALPHA = 0.2
GRPO_REWARD_THRESHOLD = 0.5
GRPO_ITERATIONS = 3
GRPO_TEMPERATURE = 1.0
GRPO_ALPHA_SWEEP = [0.0, 0.2, 0.5, 0.8, 1.0]

# ── SFT label balancing (ablation) ───────────────────────────
SFT_BALANCE_TARGET = 0.45   # minority-label share after oversampling
SFT_BALANCE_MAX_DUP = 3     # cap on duplications per example

# ── Evaluation ────────────────────────────────────────────────
# Deprecated: EN-only, legacy-keyed. New code reads expected yes-rates from
# data/splits/<lang>/cluster_meta.json (written by prepare_preference_data).
EXPECTED_YES_RATES = {
    "Conservative": 0.215,
    "Mainstream": 0.438,
    "Sensitive": 0.635,
}


# ── Legacy-artifact normalization ─────────────────────────────
def _normalize_key(key):
    """Rewrite one metric/dict key from legacy cluster naming to neutral."""
    for legacy, neutral in LEGACY_CLUSTER_TO_KEY.items():
        low = legacy.lower()
        if key == legacy:
            return neutral
        if key.endswith(f"_{low}"):
            return key[: -len(low)] + neutral
    return key


def normalize_cluster_keyed(d):
    """Return a copy of d with legacy cluster-name keys ('Conservative', ...)
    renamed to neutral keys ('cluster1', ...). Idempotent; non-dicts pass
    through unchanged. Used on cluster_votes, agent_predictions, model-id
    dicts, and similar mappings loaded from legacy JSON artifacts."""
    if not isinstance(d, dict):
        return d
    return {LEGACY_CLUSTER_TO_KEY.get(k, k): v for k, v in d.items()}


def normalize_result(result):
    """Return a copy of a result dict with legacy metric-key suffixes
    (caa_conservative, yes_rate_mainstream, calibration_sensitive, ...)
    renamed to neutral suffixes (caa_cluster1, ...). Idempotent. Nested
    dicts keyed by cluster names are normalized too."""
    if not isinstance(result, dict):
        return result
    out = {}
    for k, v in result.items():
        nk = _normalize_key(k)
        if isinstance(v, dict):
            v = normalize_cluster_keyed(v)
        out[nk] = v
    return out
