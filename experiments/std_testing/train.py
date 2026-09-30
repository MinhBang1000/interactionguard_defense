"""
Std Testing -- Phase 1: retrain Layer 1 (ModernBERT + AutoEncoder) and
Layer 2 (DeBERTa) with N different random seeds, same fixed train/val/test
split and same hyperparameters as the existing single-run training options
(main.py menu "Training and threshold tools"). Each component runs in its
own subprocess for GPU-memory isolation, mirroring main.py's existing
run_fine_tune_modermBERT / run_training_autoencoder / run_finetune_protectai.

Run this phase, then run Phase 2 (experiments/std_testing/evaluate.py)
separately once training finishes -- these are two independent invocations,
not one blocking call.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from experiments.common import ask_with_default, write_json, yes_no
from experiments.std_testing.common import (
    DEFAULT_SEEDS,
    trial_autoencoder_dir,
    trial_deberta_dir,
    trial_manifest_path,
    trial_modernbert_dir,
)
from utils.paths import MODEL_DIR, MODERNBERT_DIR

# Production checkpoints whose training_config.json records the train_ratio
# actually used to produce the reference results/mode_*_results.json numbers.
_REFERENCE_MODERNBERT_CONFIG = MODERNBERT_DIR / "training_config.json"
_REFERENCE_DEBERTA_CONFIG = (
    MODEL_DIR / "layer2" / "protectai_deberta_v3_base_prompt_injection_v2" / "training_config.json"
)

# Fallback if the production training_config.json cannot be found (e.g. fresh
# checkout without trained checkpoints yet). These mirror config/config.py's
# class defaults (ModernBERTConfig.TRAIN_RATIO, Layer2ProtectAIConfig.TRAIN_RATIO)
# only as a last resort -- always prefer the actually-used value on disk.
_FALLBACK_MODERNBERT_PERCENT = 25
_FALLBACK_DEBERTA_PERCENT = 10


def _read_reference_train_percent(config_path: Path, fallback: int) -> int:
    if not config_path.exists():
        return fallback
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ratio = data.get("train_ratio")
        if ratio is None:
            return fallback
        return round(float(ratio) * 100)
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return fallback


def _parse_seeds(raw: str) -> List[int]:
    return [int(part.strip()) for part in raw.split(",") if part.strip()]


def _resolve_cuda_visible_devices() -> str:
    """Mirror config.config.cuda_visible_devices("CUDA_DEVICE_ML", "0") without
    importing config.config (and therefore torch) into this parent process."""
    value = os.getenv("CUDA_DEVICE_ML", "0").strip()
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    return value


def _run_wrapper(args: List[str]) -> None:
    cmd = [sys.executable, "-m", "experiments.std_testing.seed_wrappers", *args]
    print(f"\n$ {' '.join(str(a) for a in cmd)}")

    # CUDA_VISIBLE_DEVICES must be set in the subprocess's environment BEFORE
    # it starts -- config/config.py calls torch.cuda.is_available() as a
    # class-attribute side effect at import time (before seed_wrappers.py
    # ever gets a chance to patch anything), which permanently locks in
    # whatever GPUs are visible at that point for the rest of the process.
    # Setting os.environ["CUDA_VISIBLE_DEVICES"] later, inside run(), is too
    # late once that has happened. This mirrors the exact pattern main.py
    # already uses for the original single-run training options
    # (run_fine_tune_modermBERT / run_training_autoencoder / run_finetune_protectai).
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = _resolve_cuda_visible_devices()
    subprocess.run(cmd, check=True, env=env)


def _train_one_trial(
    seed: int,
    index: int,
    modernbert_train_percent: int,
    deberta_train_percent: int,
    overwrite: bool,
) -> None:
    modernbert_dir = trial_modernbert_dir(seed, index)
    autoencoder_dir = trial_autoencoder_dir(seed, index)
    deberta_dir = trial_deberta_dir(seed, index)

    print(f"\n{'=' * 72}\nTrial {index} (seed={seed})\n{'=' * 72}")

    # training_config.json is written as the last step of run() on success --
    # checking for it (not just "directory has files") avoids treating a
    # partial/crashed checkpoint (e.g. an intermediate `checkpoint-N/` left
    # behind by a save that failed, such as a disk-full error) as complete.
    if (modernbert_dir / "training_config.json").exists() and not overwrite:
        print(f"  [skip] {modernbert_dir} already complete")
    else:
        _run_wrapper(
            [
                "--component", "modernbert",
                "--seed", str(seed),
                "--output_dir", str(modernbert_dir),
                "--train_percent", str(modernbert_train_percent),
            ]
        )

    if autoencoder_dir.exists() and (autoencoder_dir / "autoencoder.pth").exists() and not overwrite:
        print(f"  [skip] {autoencoder_dir / 'autoencoder.pth'} already exists")
    else:
        _run_wrapper(
            [
                "--component", "autoencoder",
                "--seed", str(seed),
                "--output_dir", str(autoencoder_dir),
                "--modernbert_dir", str(modernbert_dir),
            ]
        )

    if (deberta_dir / "training_config.json").exists() and not overwrite:
        print(f"  [skip] {deberta_dir} already complete")
    else:
        _run_wrapper(
            [
                "--component", "deberta",
                "--seed", str(seed),
                "--output_dir", str(deberta_dir),
                "--train_percent", str(deberta_train_percent),
            ]
        )

    write_json(
        trial_manifest_path(seed, index),
        {
            "trial_index": index,
            "seed": seed,
            "modernbert_train_percent": modernbert_train_percent,
            "deberta_train_percent": deberta_train_percent,
            "modernbert_dir": str(modernbert_dir),
            "autoencoder_dir": str(autoencoder_dir),
            "deberta_dir": str(deberta_dir),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(f"  Manifest written: {trial_manifest_path(seed, index)}")


def run_std_testing_train() -> None:
    print("\n" + "=" * 72)
    print("STD TESTING -- PHASE 1: MULTI-SEED L1/L2 TRAINING")
    print("=" * 72)
    print(
        "Retrains Layer 1 (ModernBERT + AutoEncoder) and Layer 2 (DeBERTa) with\n"
        "N different random seeds, keeping the fixed train/val/test split and\n"
        "every other hyperparameter identical to the existing single-run\n"
        "training options. Output goes under models/std_testing/, so the\n"
        "production checkpoints under models/layer1/ and models/layer2/ are\n"
        "never touched.\n"
    )
    print(
        "Cost warning: each trial retrains 3 models (ModernBERT, AutoEncoder,\n"
        "DeBERTa) as 3 separate GPU jobs. 5 trials = 15 training jobs total.\n"
    )

    seeds_raw = ask_with_default(
        f"Comma-separated seed list [default: {','.join(str(s) for s in DEFAULT_SEEDS)}]: ",
        ",".join(str(s) for s in DEFAULT_SEEDS),
    )
    seeds = _parse_seeds(seeds_raw)
    if not seeds:
        print("No valid seeds provided, aborting.")
        return

    # ModernBERT and DeBERTa were NOT trained on 100% of the data for the
    # existing production checkpoints -- read their actual train_ratio from
    # the training_config.json each one saved, so "same hyperparameters,
    # only seed differs" holds by default instead of silently training on a
    # different amount of data.
    modernbert_default = _read_reference_train_percent(
        _REFERENCE_MODERNBERT_CONFIG, _FALLBACK_MODERNBERT_PERCENT
    )
    deberta_default = _read_reference_train_percent(
        _REFERENCE_DEBERTA_CONFIG, _FALLBACK_DEBERTA_PERCENT
    )
    source_note = "detected from existing checkpoint's training_config.json" if _REFERENCE_MODERNBERT_CONFIG.exists() else "fallback, no reference checkpoint found"
    print(f"\nModernBERT train percent default: {modernbert_default}% ({source_note})")
    source_note = "detected from existing checkpoint's training_config.json" if _REFERENCE_DEBERTA_CONFIG.exists() else "fallback, no reference checkpoint found"
    print(f"DeBERTa train percent default    : {deberta_default}% ({source_note})")
    print(
        "These defaults match what actually produced the reference "
        "results/mode_*_results.json numbers. Only override them if you "
        "deliberately want a different training-data amount than the "
        "reference (in which case this run will no longer be directly "
        "comparable to Table 6.4).\n"
    )

    modernbert_train_percent = int(
        ask_with_default(f"ModernBERT train percent [default: {modernbert_default}]: ", str(modernbert_default))
    )
    deberta_train_percent = int(
        ask_with_default(f"DeBERTa train percent [default: {deberta_default}]: ", str(deberta_default))
    )
    overwrite = yes_no("Overwrite trials that already have output?", default=False)

    print(f"\nWill train {len(seeds)} trial(s) with seeds: {seeds}\n")
    if not yes_no("Proceed?", default=True):
        print("Aborted.")
        return

    for index, seed in enumerate(seeds, start=1):
        _train_one_trial(seed, index, modernbert_train_percent, deberta_train_percent, overwrite)

    print("\nPhase 1 complete. Run Phase 2 (Evaluate) next.")


if __name__ == "__main__":
    run_std_testing_train()
