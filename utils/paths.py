# ============================================================================
# FILE: utils/paths.py
# Project-wide path constants anchored at the repository root.
#
# Purpose:
# - Centralizes common directory references so scripts do not hardcode paths.
#
# Workflow:
# - Defines repository root.
# - Builds reusable constants for config, data, models, outputs, eval, and
#   results directories.
#
# Use this file when:
# - You need to understand where scripts read from or write to by default.
# ============================================================================
# utils/paths.py

from pathlib import Path

# <Project Root>/utils/paths.py -> Target to <Project Root>
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Configs
CONFIG_DIR = PROJECT_ROOT / "config"

# Data Dir
DATA_DIR = PROJECT_ROOT / "data"
SPLIT_DIR = DATA_DIR / "splits"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTED_DIR = DATA_DIR / "predicted"

# Docs
DOCS_DIR = PROJECT_ROOT / "docs"

# Models
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_LAYER1_DIR = MODEL_DIR / "layer1"
MODERNBERT_DIR = MODEL_LAYER1_DIR / "answerdotai_ModernBERT-base_finetuned"
DETECTOR_DIR = MODEL_DIR / "detectors"

# Eval
EVAL_DIR = PROJECT_ROOT / "eval"

# Outputs
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LAYER1_DIR = OUTPUT_DIR / "layer1"
STAGE1_DIR = LAYER1_DIR / "stage1"
STAGE2_DIR = LAYER1_DIR / "stage2"

# Results
RESULT_DIR = PROJECT_ROOT / "results"
