"""
Shared constants and path helpers for the Std Testing feature.

Std Testing repeats Layer 1 / Layer 2 training with different random seeds
(same fixed data split, same hyperparameters) and re-evaluates the 9 modes
that make up the thesis's Table 6.4, so mean +/- std can be reported instead
of a single run's point estimate.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

from utils.paths import CONFIG_DIR, MODEL_DIR, RESULT_DIR

DEFAULT_SEEDS: List[int] = [42, 43, 44, 45, 46]

# The 9 modes that make up thesis Table 6.4 ("Layer configuration comparison
# across evaluated configurations"), in config/test_modes.yaml.
TABLE_6_4_MODES: List[int] = [1, 2, 3, 10, 13, 14, 15, 11, 12]

# Reference values verified to reproduce Table 6.4 exactly.
# 2026-09-17: corrected from 0.01 -> 0.05. 0.01 was a leftover value from a
# different experiment; with the freshly-recomputed Layer 1 embeddings,
# 0.05 is the threshold that reproduces the Table 6.4 Layer 1 numbers
# (precision=0.7911, recall=0.9915, fpr=0.2577, verified against
# results/mode_1_results.json).
REFERENCE_LAYER1_THRESHOLD = 0.05
REFERENCE_LAYER2_THRESHOLD = 0.5
REFERENCE_LAYER3_VARIANT = "o2"

STD_TESTING_MODEL_DIR = MODEL_DIR / "std_testing"
STD_TESTING_RESULT_DIR = RESULT_DIR / "std_testing"
STD_TESTING_CONFIG_DIR = CONFIG_DIR / "std_testing"


def trial_name(seed: int, index: int) -> str:
    return f"trial{index}_seed{seed}"


def trial_model_dir(seed: int, index: int) -> Path:
    return STD_TESTING_MODEL_DIR / trial_name(seed, index)


def trial_modernbert_dir(seed: int, index: int) -> Path:
    return trial_model_dir(seed, index) / "layer1_modernbert"


def trial_autoencoder_dir(seed: int, index: int) -> Path:
    return trial_model_dir(seed, index) / "layer1_autoencoder"


def trial_deberta_dir(seed: int, index: int) -> Path:
    return trial_model_dir(seed, index) / "layer2_deberta"


def trial_manifest_path(seed: int, index: int) -> Path:
    return trial_model_dir(seed, index) / "manifest.json"


def trial_result_dir(seed: int, index: int) -> Path:
    return STD_TESTING_RESULT_DIR / trial_name(seed, index)


def trial_config_path(seed: int, index: int) -> Path:
    return STD_TESTING_CONFIG_DIR / f"{trial_name(seed, index)}_detection_config.yaml"
