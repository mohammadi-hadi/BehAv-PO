#!/usr/bin/env python3
"""
OpenAI Spanish Suite Orchestrator (resume-safe)
================================================
Runs the full Spanish OpenAI pipeline on gpt-4.1-mini-2025-04-14:

  0. en_backfill    EN predictions backfill for the frozen EN experiments
                    (predictions only; existing EN results are NOT touched)
  1. sft_create     upload ES SFT JSONLs + create 3 fine-tuning jobs
  2. sft_wait       poll -> artifacts/model_ids/es_sft_41mini_model_ids.json
  3. dpo_create     DPO jobs on top of each ES SFT model
  4. dpo_wait       poll -> artifacts/model_ids/es_dpo_41mini_model_ids.json
  5. grpo_sample    rejection sampling (convex reward) -> grpo_a20 JSONLs
  6. grpo_create    fine-tune from each ES SFT model on the filtered samples
  7. grpo_wait      poll -> artifacts/model_ids/es_grpo_a20_41mini_model_ids.json
  8. predict_results ES predictions + results for sft / dpo / grpo_a20 stages

State lives in artifacts/openai_es_state.json ({step: {status, ids...}}).
Every step is idempotent: it is skipped when the state says done and the
step's artifacts exist, and job-creation steps skip clusters whose job ids
are already recorded, so the script can be re-run safely after interruption.

Usage:
    python3 src/training/openai_es_suite.py            # run all pending steps
    python3 src/training/openai_es_suite.py --only sft_wait
    python3 src/training/openai_es_suite.py --status

API key: read from the OPENAI_API_KEY environment variable, falling back to
parsing the `.env` file at the repo root (line `OPENAI_API_KEY=...`).
The key is never printed and never written to any file.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402

LANG = "es"
ES_BASE_MODEL = "gpt-4.1-mini-2025-04-14"
ES_PREF_DIR = config.PREFERENCE_DIR / "es"
STATE_PATH = config.ARTIFACTS_DIR / "openai_es_state.json"
PREDICT_SCRIPT = config.REPO_ROOT / "src" / "evaluation" / "openai_predict.py"

GRPO_SAMPLE_CAP = 1000        # max train texts used for rejection sampling
GRPO_SAMPLE_WORKERS = 8       # one in-flight API call per worker
POLL_INTERVAL = 60            # seconds between fine-tuning job polls
MAX_RETRIES = 3               # per-request retries (exponential backoff)

MODEL_IDS_FILES = {
    "sft": config.MODEL_IDS_DIR / "es_sft_41mini_model_ids.json",
    "dpo": config.MODEL_IDS_DIR / "es_dpo_41mini_model_ids.json",
    "grpo": config.MODEL_IDS_DIR / "es_grpo_a20_41mini_model_ids.json",
}

# EN experiments to backfill predictions for (results already exist and are
# intentionally left untouched -> --no-results).
EN_BACKFILL_KEYS = [
    "sft_41mini",
    "dpo_41mini",
    "marspo_41mini",
    "grpo_41mini_alpha02",
    "sft_pure_41mini",
]

# ES stages -> prediction/result labels.
ES_PREDICT_RUNS = [
    ("sft", "gpt41mini_sft_es"),
    ("dpo", "gpt41mini_dpo_es"),
    ("grpo", "gpt41mini_grpo_a20_es"),
]

STEPS = [
    "en_backfill",
    "sft_create", "sft_wait",
    "dpo_create", "dpo_wait",
    "grpo_sample", "grpo_create", "grpo_wait",
    "predict_results",
]


# ── Small helpers ─────────────────────────────────────────────

def ts():
    return datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")


def load_api_key():
    """OPENAI_API_KEY from env, falling back to `.env` at the repo root."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    env_path = config.REPO_ROOT / ".env"
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("export "):
                    line = line[len("export "):].strip()
                if line.startswith("OPENAI_API_KEY"):
                    _, _, val = line.partition("=")
                    val = val.strip().strip('"').strip("'")
                    if val:
                        return val
    raise SystemExit(
        "OPENAI_API_KEY not found in the environment or in "
        f"{env_path} (expected line: OPENAI_API_KEY=...)"
    )


def load_state():
    if STATE_PATH.exists():
        with open(STATE_PATH) as f:
            return json.load(f)
    return {}


def save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def step_done(state, step):
    return state.get(step, {}).get("status") == "done"


def mark_done(state, step, **extra):
    entry = state.setdefault(step, {})
    entry["status"] = "done"
    entry["finished_at"] = ts()
    entry.update(extra)
    save_state(state)


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def upload_file(client, filepath):
    print(f"{ts()}   uploading {filepath} ...", flush=True)
    with open(filepath, "rb") as f:
        resp = client.files.create(file=f, purpose="fine-tune")
    print(f"{ts()}   file id: {resp.id}", flush=True)
    return resp.id


def get_job(client, job_id):
    """jobs.retrieve with a jobs.list fallback (retrieve occasionally 401s)."""
    try:
        return client.fine_tuning.jobs.retrieve(job_id)
    except Exception as e:
        print(f"{ts()}   retrieve({job_id}) failed ({type(e).__name__}); "
              f"falling back to jobs.list", flush=True)
        for job in client.fine_tuning.jobs.list(limit=50).data:
            if job.id == job_id:
                return job
        raise


def load_model_ids_file(key):
    path = MODEL_IDS_FILES[key]
    if not path.exists():
        raise SystemExit(f"{path} not found; run the corresponding wait step first.")
    with open(path) as f:
        model_ids = config.normalize_cluster_keyed(json.load(f))
    missing = [ck for ck in config.CLUSTER_KEYS if not model_ids.get(ck)]
    if missing:
        raise SystemExit(f"{path} is missing model ids for: {missing}")
    return model_ids


def model_ids_ready(key):
    path = MODEL_IDS_FILES[key]
    if not path.exists():
        return False
    try:
        with open(path) as f:
            ids = config.normalize_cluster_keyed(json.load(f))
        return all(ids.get(ck) for ck in config.CLUSTER_KEYS)
    except (ValueError, OSError):
        return False


def run_predict(model_ids_path, lang, label, write_results):
    """Shell out to openai_predict.py with the same interpreter."""
    cmd = [sys.executable, str(PREDICT_SCRIPT),
           "--model-ids", str(model_ids_path),
           "--lang", lang, "--label", label]
    if not write_results:
        cmd.append("--no-results")
    env = dict(os.environ)
    env["OPENAI_API_KEY"] = load_api_key()
    env.setdefault("PYTHONUNBUFFERED", "1")
    print(f"{ts()} $ {sys.executable} {PREDICT_SCRIPT} --model-ids "
          f"{model_ids_path} --lang {lang} --label {label}"
          f"{' --no-results' if not write_results else ''}", flush=True)
    proc = subprocess.run(cmd, cwd=str(config.REPO_ROOT), env=env)
    if proc.returncode != 0:
        raise RuntimeError(f"openai_predict.py failed for label={label} "
                           f"(exit {proc.returncode})")


# ── Step 0: EN predictions backfill ───────────────────────────

def step_en_backfill(client, state):
    entry = state.setdefault("en_backfill", {"status": "pending", "runs": {}})
    for key in EN_BACKFILL_KEYS:
        pred_path = config.PREDICTIONS_DIR / f"{key}_predictions.json"
        if entry["runs"].get(key) == "done" and pred_path.exists():
            print(f"{ts()} en_backfill: {key} already done, skipping", flush=True)
            continue
        ids_path = config.MODEL_IDS_DIR / f"{key}_model_ids.json"
        if not ids_path.exists():
            print(f"{ts()} en_backfill: {ids_path} missing, skipping {key}",
                  flush=True)
            entry["runs"][key] = "missing_model_ids"
            save_state(state)
            continue
        # Predictions only: EN results already exist and must not be rewritten.
        run_predict(ids_path, "en", key, write_results=False)
        entry["runs"][key] = "done"
        save_state(state)
    mark_done(state, "en_backfill")


# ── Fine-tuning job creation / waiting ────────────────────────

def create_jobs(client, state, step, job_builder):
    """Create one fine-tuning job per cluster, resume-safe per cluster.

    job_builder(client, ck, entry) -> job params dict for jobs.create().
    """
    entry = state.setdefault(step, {"status": "pending", "jobs": {}, "files": {}})
    entry.setdefault("jobs", {})
    entry.setdefault("files", {})
    for ck in config.CLUSTER_KEYS:
        if entry["jobs"].get(ck):
            print(f"{ts()} {step}: {ck} job already created "
                  f"({entry['jobs'][ck]}), skipping", flush=True)
            continue
        params = job_builder(client, ck, entry)
        job = client.fine_tuning.jobs.create(**params)
        print(f"{ts()} {step}: {ck} job id {job.id} (status {job.status})",
              flush=True)
        entry["jobs"][ck] = job.id
        save_state(state)
    mark_done(state, step)


def wait_for_jobs(client, state, wait_step, create_step, out_path):
    """Poll every POLL_INTERVAL s until all 3 jobs succeed; save model ids."""
    if model_ids_ready(_key_for(out_path)):
        print(f"{ts()} {wait_step}: {out_path} already complete, skipping",
              flush=True)
        mark_done(state, wait_step)
        return
    jobs = state.get(create_step, {}).get("jobs", {})
    missing = [ck for ck in config.CLUSTER_KEYS if not jobs.get(ck)]
    if missing:
        raise SystemExit(f"{wait_step}: no job ids recorded for {missing}; "
                         f"run {create_step} first.")
    model_ids = {}
    while True:
        statuses = {}
        for ck in config.CLUSTER_KEYS:
            if ck in model_ids:
                statuses[ck] = "succeeded"
                continue
            job = get_job(client, jobs[ck])
            statuses[ck] = job.status
            if job.status == "succeeded":
                model_ids[ck] = job.fine_tuned_model
            elif job.status in ("failed", "cancelled"):
                raise RuntimeError(
                    f"{wait_step}: job {jobs[ck]} ({ck}) {job.status}: "
                    f"{getattr(job, 'error', None)}")
        print(f"{ts()} {wait_step}: " +
              " ".join(f"{ck}={statuses[ck]}" for ck in config.CLUSTER_KEYS),
              flush=True)
        if len(model_ids) == len(config.CLUSTER_KEYS):
            break
        time.sleep(POLL_INTERVAL)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({ck: model_ids[ck] for ck in config.CLUSTER_KEYS}, f, indent=2)
    print(f"{ts()} {wait_step}: saved {out_path}", flush=True)
    mark_done(state, wait_step, model_ids=model_ids)


def _key_for(out_path):
    for key, path in MODEL_IDS_FILES.items():
        if path == out_path:
            return key
    raise KeyError(out_path)


# ── Steps 1-2: SFT ────────────────────────────────────────────

def step_sft_create(client, state):
    def builder(client, ck, entry):
        train_path = ES_PREF_DIR / f"sft_{ck}_train.jsonl"
        val_path = ES_PREF_DIR / f"sft_{ck}_val.jsonl"
        print(f"{ts()} sft_create: {ck} train={count_lines(train_path)} lines, "
              f"val={count_lines(val_path)} lines", flush=True)
        train_id = upload_file(client, train_path)
        val_id = upload_file(client, val_path)
        entry["files"][ck] = {"train": train_id, "val": val_id}
        return {
            "training_file": train_id,
            "validation_file": val_id,
            "model": ES_BASE_MODEL,
            "hyperparameters": {"n_epochs": config.SFT_EPOCHS},
            "suffix": f"behav-po-es-sft-{ck}",
        }
    create_jobs(client, state, "sft_create", builder)


def step_sft_wait(client, state):
    wait_for_jobs(client, state, "sft_wait", "sft_create",
                  MODEL_IDS_FILES["sft"])


# ── Steps 3-4: DPO (on top of ES SFT models) ─────────────────

def step_dpo_create(client, state):
    sft_ids = load_model_ids_file("sft")

    def builder(client, ck, entry):
        train_path = ES_PREF_DIR / f"dpo_{ck}_train.jsonl"
        val_path = ES_PREF_DIR / f"dpo_{ck}_val.jsonl"
        print(f"{ts()} dpo_create: {ck} train={count_lines(train_path)} lines, "
              f"val={count_lines(val_path)} lines "
              f"(base {sft_ids[ck]})", flush=True)
        train_id = upload_file(client, train_path)
        val_id = upload_file(client, val_path)
        entry["files"][ck] = {"train": train_id, "val": val_id}
        return {
            "training_file": train_id,
            "validation_file": val_id,
            "model": sft_ids[ck],
            # Method-dict shape as in src/training/dpo_finetune.py.
            "method": {
                "type": "dpo",
                "dpo": {
                    "hyperparameters": {
                        "beta": config.DPO_BETA,
                        "n_epochs": 2,
                    },
                },
            },
            "suffix": f"behav-po-es-dpo-{ck}",
        }
    create_jobs(client, state, "dpo_create", builder)


def step_dpo_wait(client, state):
    wait_for_jobs(client, state, "dpo_wait", "dpo_create",
                  MODEL_IDS_FILES["dpo"])


# ── Step 5: GRPO rejection sampling ───────────────────────────

def grpo_train_path(ck):
    return ES_PREF_DIR / f"grpo_a20_{ck}_train.jsonl"


def sample_agent_text(client, model_id, tweet):
    """One API call returning all K sampled YES/NO labels for one text."""
    messages = [
        {"role": "system", "content": config.SYSTEM_PROMPT},
        {"role": "user", "content": f"Tweet: {tweet}"},
    ]
    delay = 1.0
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=messages,
                n=config.GRPO_K_SAMPLES,
                temperature=config.GRPO_TEMPERATURE,
                max_tokens=3,
            )
            labels = []
            for choice in resp.choices:
                raw = (choice.message.content or "").strip().upper()
                labels.append("YES" if "YES" in raw else "NO")
            return labels
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise
            print(f"{ts()}   retry {attempt + 1}/{MAX_RETRIES} "
                  f"after {type(e).__name__}: {e}", flush=True)
            time.sleep(delay)
            delay *= 2


def step_grpo_sample(client, state):
    out_paths = {ck: grpo_train_path(ck) for ck in config.CLUSTER_KEYS}
    if step_done(state, "grpo_sample") and all(
            p.exists() for p in out_paths.values()):
        print(f"{ts()} grpo_sample: outputs exist, skipping", flush=True)
        return

    sft_ids = load_model_ids_file("sft")
    with open(config.SPLITS_DIR / LANG / "train_set.json") as f:
        train_data = json.load(f)
    train_data = random.Random(config.RANDOM_STATE).sample(
        train_data, min(GRPO_SAMPLE_CAP, len(train_data)))
    n_texts = len(train_data)
    n_calls = n_texts * len(config.CLUSTER_KEYS)
    print(f"{ts()} grpo_sample: {n_texts} texts x {len(config.CLUSTER_KEYS)} "
          f"agents = {n_calls} API calls, K={config.GRPO_K_SAMPLES}, "
          f"alpha={config.GRPO_ALPHA}, threshold={config.GRPO_REWARD_THRESHOLD}",
          flush=True)

    tweets = [r["tweet"].replace("\x00", "") for r in train_data]

    # samples[ck][i] = list of K YES/NO labels
    samples = {ck: [None] * n_texts for ck in config.CLUSTER_KEYS}
    tasks = [(ck, i) for ck in config.CLUSTER_KEYS for i in range(n_texts)]
    with ThreadPoolExecutor(max_workers=GRPO_SAMPLE_WORKERS) as ex:
        futures = {
            ex.submit(sample_agent_text, client, sft_ids[ck], tweets[i]): (ck, i)
            for ck, i in tasks
        }
        done = 0
        for fut in as_completed(futures):
            ck, i = futures[fut]
            samples[ck][i] = fut.result()
            done += 1
            if done % 100 == 0 or done == n_calls:
                print(f"{ts()} grpo_sample: {done}/{n_calls} calls done",
                      flush=True)

    # Fixed per-agent majority-of-K label per text (ties -> NO).
    majority = {
        ck: ["YES" if sum(1 for s in samples[ck][i] if s == "YES")
             > config.GRPO_K_SAMPLES / 2 else "NO"
             for i in range(n_texts)]
        for ck in config.CLUSTER_KEYS
    }

    alpha = config.GRPO_ALPHA
    counts = {}
    for ck in config.CLUSTER_KEYS:
        kept = {}
        n_total = n_above = 0
        for i, record in enumerate(train_data):
            cv = record.get("cluster_votes", {}).get(ck)
            cluster_gold = cv.get("majority") if cv else None
            overall = record.get("overall_majority")
            others = [ok for ok in config.CLUSTER_KEYS if ok != ck]
            for label in samples[ck][i]:
                n_total += 1
                r_indiv = 1.0 if (cluster_gold is not None
                                  and label == cluster_gold) else 0.0
                # Team vote per sample: other agents' majority-of-K + sample.
                votes = [majority[ok][i] for ok in others] + [label]
                team_vote = ("YES" if sum(1 for v in votes if v == "YES") >= 2
                             else "NO")
                r_team = 1.0 if (overall is not None
                                 and team_vote == overall) else 0.0
                reward = (1 - alpha) * r_indiv + alpha * r_team
                if reward > config.GRPO_REWARD_THRESHOLD:
                    n_above += 1
                    key = (tweets[i], label)
                    if key not in kept or reward > kept[key]:
                        kept[key] = reward

        out_path = out_paths[ck]
        with open(out_path, "w") as f:
            for (tweet, label), _reward in kept.items():
                f.write(json.dumps({
                    "messages": [
                        {"role": "system", "content": config.SYSTEM_PROMPT},
                        {"role": "user", "content": f"Tweet: {tweet}"},
                        {"role": "assistant", "content": label},
                    ]
                }, ensure_ascii=False) + "\n")
        counts[ck] = len(kept)
        print(f"{ts()} grpo_sample: {ck}: {n_total} samples -> "
              f"{n_above} above threshold -> {len(kept)} unique -> {out_path}",
              flush=True)

    mark_done(state, "grpo_sample", n_texts=n_texts, kept=counts)


# ── Steps 6-7: GRPO fine-tune (from ES SFT models) ────────────

def step_grpo_create(client, state):
    sft_ids = load_model_ids_file("sft")

    def builder(client, ck, entry):
        train_path = grpo_train_path(ck)
        if not train_path.exists():
            raise SystemExit(f"{train_path} not found; run grpo_sample first.")
        val_path = ES_PREF_DIR / f"sft_{ck}_val.jsonl"  # reuse SFT val set
        print(f"{ts()} grpo_create: {ck} train={count_lines(train_path)} lines, "
              f"val={count_lines(val_path)} lines (base {sft_ids[ck]})",
              flush=True)
        train_id = upload_file(client, train_path)
        # Reuse the SFT val file id when we still have it, else re-upload.
        val_id = (state.get("sft_create", {}).get("files", {})
                  .get(ck, {}).get("val"))
        if not val_id:
            val_id = upload_file(client, val_path)
        entry["files"][ck] = {"train": train_id, "val": val_id}
        return {
            "training_file": train_id,
            "validation_file": val_id,
            "model": sft_ids[ck],
            "hyperparameters": {"n_epochs": 1},
            "suffix": f"behav-po-es-grpo-a20-{ck}",
        }
    create_jobs(client, state, "grpo_create", builder)


def step_grpo_wait(client, state):
    wait_for_jobs(client, state, "grpo_wait", "grpo_create",
                  MODEL_IDS_FILES["grpo"])


# ── Step 8: ES predictions + results ──────────────────────────

def step_predict_results(client, state):
    entry = state.setdefault("predict_results", {"status": "pending", "runs": {}})
    for stage_key, label in ES_PREDICT_RUNS:
        pred_path = config.PREDICTIONS_DIR / f"{label}_predictions.json"
        res_path = config.RESULTS_DIR / f"{label}_results.json"
        if (entry["runs"].get(label) == "done"
                and pred_path.exists() and res_path.exists()):
            print(f"{ts()} predict_results: {label} already done, skipping",
                  flush=True)
            continue
        run_predict(MODEL_IDS_FILES[stage_key], LANG, label,
                    write_results=True)
        entry["runs"][label] = "done"
        save_state(state)
    mark_done(state, "predict_results")


# ── Driver ────────────────────────────────────────────────────

STEP_FUNCS = {
    "en_backfill": step_en_backfill,
    "sft_create": step_sft_create,
    "sft_wait": step_sft_wait,
    "dpo_create": step_dpo_create,
    "dpo_wait": step_dpo_wait,
    "grpo_sample": step_grpo_sample,
    "grpo_create": step_grpo_create,
    "grpo_wait": step_grpo_wait,
    "predict_results": step_predict_results,
}


def print_status(state):
    print(f"State file: {STATE_PATH}"
          f" ({'exists' if STATE_PATH.exists() else 'not created yet'})")
    for step in STEPS:
        entry = state.get(step, {})
        status = entry.get("status", "pending")
        extra = ""
        if step in ("sft_create", "dpo_create", "grpo_create"):
            jobs = entry.get("jobs", {})
            extra = f" jobs={len(jobs)}/{len(config.CLUSTER_KEYS)}"
        elif step.endswith("_wait"):
            key = step.split("_")[0]
            extra = (f" model_ids={'ready' if model_ids_ready(key) else 'missing'}"
                     f" ({MODEL_IDS_FILES[key].name})")
        elif step == "grpo_sample":
            n = sum(1 for ck in config.CLUSTER_KEYS if grpo_train_path(ck).exists())
            extra = f" jsonl={n}/{len(config.CLUSTER_KEYS)}"
        elif step == "en_backfill":
            runs = entry.get("runs", {})
            extra = (f" runs={sum(1 for v in runs.values() if v == 'done')}"
                     f"/{len(EN_BACKFILL_KEYS)}")
        elif step == "predict_results":
            runs = entry.get("runs", {})
            extra = (f" runs={sum(1 for v in runs.values() if v == 'done')}"
                     f"/{len(ES_PREDICT_RUNS)}")
        print(f"  {step:<16} {status}{extra}")


def main():
    parser = argparse.ArgumentParser(
        description="Resume-safe Spanish OpenAI suite (gpt-4.1-mini)")
    parser.add_argument("--only", choices=STEPS,
                        help="Run a single step instead of all pending steps")
    parser.add_argument("--status", action="store_true",
                        help="Print state summary and exit")
    args = parser.parse_args()

    state = load_state()

    if args.status:
        print_status(state)
        return

    client = OpenAI(api_key=load_api_key())
    steps = [args.only] if args.only else STEPS

    for step in steps:
        if step_done(state, step) and not args.only:
            print(f"{ts()} {step}: done, skipping", flush=True)
            continue
        print(f"{ts()} ===== {step} =====", flush=True)
        STEP_FUNCS[step](client, state)

    print(f"{ts()} all requested steps complete.", flush=True)
    print_status(load_state())


if __name__ == "__main__":
    main()
