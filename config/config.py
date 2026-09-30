# ============================================================================
# FILE: config/config.py
# Central Python configuration registry for training, preprocessing, and splits.
#
# Purpose:
# - Defines strongly grouped config classes for Layer 1, Layer 2, data
#   preprocessing, and dataset splitting.
#
# Workflow:
# - Training scripts import model-specific config classes from here.
# - Data scripts import split and preprocessing settings from here.
# - Paths and hyperparameters are kept centralized to reduce duplication.
#
# Use this file when:
# - You need to change training hyperparameters, subset ratios, model names,
#   max sequence lengths, or default output locations.
# ============================================================================
from pathlib import Path
import os
import torch
from dotenv import load_dotenv
from utils.random_seed import SEED
from utils.paths import RAW_DIR, SPLIT_DIR, PROCESSED_DIR, MODERNBERT_DIR, MODEL_LAYER1_DIR, STAGE2_DIR

import torch
from pathlib import Path
from utils.paths import PROCESSED_DIR

load_dotenv()


def cuda_visible_devices(env_name: str = "CUDA_DEVICE_ML", default: str = "0") -> str:
    value = os.getenv(env_name, default)
    value = str(value or default).strip()
    if value.startswith("cuda:"):
        return value.split(":", 1)[1]
    return value


def torch_device_from_env(env_name: str = "CUDA_DEVICE_ML", default: str = "0") -> str:
    value = os.getenv(env_name, default)
    value = str(value or default).strip().lower()
    if value in {"cpu", "none", "false", "-1"}:
        return "cpu"
    if value.startswith("cuda"):
        return value
    first = value.split(",", 1)[0].strip()
    return f"cuda:{first}" if first else "cuda"


class Layer2SentinelConfig:

    # ================================
    # Model
    # ================================
    MODEL_NAME = "qualifire/prompt-injection-sentinel"
    NUM_LABELS = 2
    TRUST_REMOTE_CODE = True

    # Sentinel backbone = ModernBERT-large
    MAX_LENGTH = 2048


    # ================================
    # Data
    # ================================
    DATA_DIR = PROCESSED_DIR

    TRAIN_FILE = "train_processed.jsonl"
    VAL_FILE = "val_processed.jsonl"

    # only use subset to prevent overfitting
    TRAIN_RATIO = 0.25


    # ================================
    # Output
    # ================================
    OUTPUT_DIR = Path("models/layer2/sentinel_prompt_injection")


    # ================================
    # Training
    # ================================
    EPOCHS = 3

    LR = 2e-5

    BATCH_SIZE = 4
    GRAD_ACCUM = 8

    # effective batch = 32


    # ================================
    # Optimization
    # ================================
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0


    # ================================
    # Evaluation
    # ================================
    EVAL_STRATEGY = "steps"

    EVAL_STEPS = 500

    SAVE_STEPS = 500
    SAVE_TOTAL_LIMIT = 2

    METRIC_FOR_BEST_MODEL = "f1"
    GREATER_IS_BETTER = True


    # ================================
    # Early Stopping
    # ================================
    EARLY_STOP_PATIENCE = 2


    # ================================
    # Logging
    # ================================
    LOGGING_STEPS = 100
    REPORT_TO = "none"


    # ================================
    # System
    # ================================
    SEED = 42

    DATALOADER_NUM_WORKERS = 4
    PIN_MEMORY = True
    DROP_LAST = False


    # ================================
    # Mixed Precision
    # ================================
    FP16 = torch.cuda.is_available()


    # ================================
    # Runtime device
    # ================================
    DEVICE = torch_device_from_env() if torch.cuda.is_available() else "cpu"

class Layer2ProtectAIConfig:
    # -------------------------
    # Model
    # -------------------------
    # Default: easier practical choice for Layer 2
    MODEL_NAME = "protectai/deberta-v3-base-prompt-injection-v2"

    # If later you want Sentinel-style backbone instead, replace with:
    # MODEL_NAME = "qualifire/prompt-injection-sentinel"

    NUM_LABELS = 2
    TRUST_REMOTE_CODE = True

    # -------------------------
    # Paths
    # -------------------------
    DATA_DIR = Path("data/processed")
    TRAIN_FILE = "train_processed.jsonl"
    VAL_FILE = "val_processed.jsonl"

    OUTPUT_DIR = Path("models/layer2/protectai_deberta_v3_base_prompt_injection_v2")

    # -------------------------
    # Data / Tokenization
    # -------------------------
    MAX_LENGTH = 512 # Edit this
    TRAIN_RATIO = 0.10 # Edit this 0.25 standard
    SEED = 42
    CUDA_VISIBLE_DEVICES = cuda_visible_devices("CUDA_DEVICE_ML", "0")

    # -------------------------
    # Optimization
    # -------------------------
    EPOCHS = 3
    LR = 2e-5
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    BATCH_SIZE = 16
    GRAD_ACCUM = 2

    # -------------------------
    # Eval / Save / Logging
    # -------------------------
    EVAL_STEPS = 200
    LOGGING_STEPS = 50
    SAVE_TOTAL_LIMIT = 2
    EARLY_STOP_PATIENCE = 2

    # -------------------------
    # Runtime
    # -------------------------
    DATALOADER_NUM_WORKERS = 4
    FP16 = torch.cuda.is_available()


class Layer3Llama31SFTConfig:
    # ================================
    # Model
    # ================================
    MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
    OLLAMA_MODEL_NAME = "llama3.1:8b"

    # ================================
    # Data
    # ================================
    DATA_DIR = PROCESSED_DIR
    TRAIN_FILE = "train_processed.jsonl"
    VAL_FILE = "val_processed.jsonl"

    # ================================
    # Output
    # ================================
    OUTPUT_ROOT = Path("models/layer3")

    # ================================
    # Sequence / Prompt
    # ================================
    MAX_LENGTH = 1024
    MAX_NEW_TOKENS = 96

    # ================================
    # QLoRA
    # ================================
    USE_QLORA = True
    LOAD_IN_4BIT = True
    BNB_4BIT_QUANT_TYPE = "nf4"
    BNB_4BIT_USE_DOUBLE_QUANT = True
    OLLAMA_STOP_TOKENS = [
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    # ================================
    # Training
    # ================================
    EPOCHS = 2
    LR_LORA = 2e-4
    LR_FULL = 2e-5
    BATCH_SIZE = 1
    EVAL_BATCH_SIZE = 1
    GRAD_ACCUM = 16

    # ================================
    # Optimization
    # ================================
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.03
    MAX_GRAD_NORM = 1.0
    USE_GRADIENT_CHECKPOINTING = True

    # ================================
    # LoRA
    # ================================
    LORA_R = 16
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    # ================================
    # Eval / Save / Logging
    # ================================
    EVAL_STEPS = 100
    LOGGING_STEPS = 10
    SAVE_TOTAL_LIMIT = 2

    # ================================
    # Runtime
    # ================================
    SEED = 42
    DATALOADER_NUM_WORKERS = 2
    REPORT_TO = "none"
    CUDA_VISIBLE_DEVICES = cuda_visible_devices(
        "CUDA_DEVICE_LLM",
        cuda_visible_devices("CUDA_DEVICE_ML", "0"),
    )

class AutoEncoderConfig:
    """Configuration for Autoencoder training (Stage 2)"""

    CUDA_VISIBLE_DEVICES = cuda_visible_devices("CUDA_DEVICE_ML", "0")

    # === Paths ===
    PROCESSED_DIR = PROCESSED_DIR
    MODEL_DIR = MODEL_LAYER1_DIR
    OUTPUT_DIR = STAGE2_DIR

    # === Data files ===
    TRAIN_FILE = "train_processed.jsonl"
    VAL_FILE = "val_processed.jsonl"
    TEST_FILE = "test_processed.jsonl"

    # === Fine-tuned embedding model (Stage 1 output) ===
    FINETUNED_MODEL_PATH = MODERNBERT_DIR

    def __init__(self):

        # Load Stage 1 training configuration
        self.BERT_MODEL_NAME = "answerdotai/ModernBERT-base"
        self.MAX_LENGTH = 2048
        self.TRUST_REMOTE_CODE = True

        # === Autoencoder Architecture ===
        self.EMBEDDING_DIM = 768
        self.HIDDEN_DIMS = [512, 256, 128]
        self.LATENT_DIM = 64
        self.DROPOUT = 0.1

        # === Training ===
        self.EPOCHS = 50
        self.BATCH_SIZE = 128
        self.LEARNING_RATE = 1e-3
        self.WEIGHT_DECAY = 1e-5

        # === Early stopping ===
        self.PATIENCE = 10
        self.MIN_DELTA = 1e-4

        # === Random seed ===
        self.SEED = 42

        # Ensure output directory exists
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

class ModernBERTConfig:

    # ================================
    # Model
    # ================================
    MODEL_NAME = "answerdotai/ModernBERT-base"
    MAX_LENGTH = 2048

    # ================================
    # Data
    # ================================
    DATA_DIR = PROCESSED_DIR
    TRAIN_FILE = "train_processed.jsonl"
    VAL_FILE = "val_processed.jsonl"

    OUTPUT_DIR = MODERNBERT_DIR

    TRAIN_RATIO = 0.25   # control training data amount # good at 25%
    CUDA_VISIBLE_DEVICES = cuda_visible_devices("CUDA_DEVICE_ML", "0")

    # ================================
    # Training
    # ================================
    EPOCHS = 3
    LR = 2e-5

    BATCH_SIZE = 4
    GRAD_ACCUM = 8

    # ================================
    # Optimization
    # ================================
    WEIGHT_DECAY = 0.01
    WARMUP_RATIO = 0.1
    MAX_GRAD_NORM = 1.0

    # ================================
    # Evaluation
    # ================================
    EVAL_STRATEGY = "steps"
    EVAL_STEPS = 500

    SAVE_STEPS = 500
    SAVE_TOTAL_LIMIT = 2

    METRIC_FOR_BEST_MODEL = "f1"
    GREATER_IS_BETTER = True

    # ================================
    # Early stopping
    # ================================
    EARLY_STOP_PATIENCE = 2

    # ================================
    # Logging
    # ================================
    LOGGING_STEPS = 100
    REPORT_TO = "none"

    # ================================
    # System
    # ================================
    SEED = 42
    NUM_WORKERS = 4
    PIN_MEMORY = True
    DROP_LAST = False

    # ================================
    # Mixed precision
    # ================================
    FP16 = True



class DataPreprocessConfig:
    # path
    SPLIT_DIR = SPLIT_DIR
    PROCESSED_DIR = PROCESSED_DIR

    # input files to process
    INPUT_FILES = [
        "train.jsonl",
        "val.jsonl",
        "test.jsonl"
    ]

    # preprocessing options
    """
        - Use multi-step for training (it looks like data augmentation)
    """
    GENERATE_MULTI_STEP = {
        "train": True,
        "val": True,
        "test": True
    }

    # text normalization
    BASIC_NORMALIZATION_ONLY = True

    # special tokens
    SPECIAL_TOKENS = {
        "prompt": "[PROMPT]",
        "retrieve": "[RETRIEVE]",
        "agent": "[AGENT]",
        "tool": "[TOOL: {name}]",
        "sep": "[SEP]",
        "eos": "[EOS]"
    }

    def __post_init__(self):
        self.PROCESSED_DIR.mkdir(exist_ok=True)

        # File not found validation
        for fname in self.INPUT_FILES:
            fpath = self.SPLIT_DIR / fname
            if not fpath.exists():
                raise FileNotFoundError(f"Could not find this {fname} file in {self.SPLIT_DIR}")

class DataSplitConfig:
    # Path
    LOG_DIR = RAW_DIR
    SPLIT_DIR = SPLIT_DIR

    # Random Seed
    SEED = SEED

    # Ratios
    BENIGN_TRAIN_RATIO = 0.6
    BENIGN_VAL_RATIO = 0.2
    BENIGN_TEST_RATIO = 0.2

    ATTACK_TRAIN_RATIO = 0.5
    ATTACK_VAL_RATIO = 0.2
    ATTACK_TEST_RATIO = 0.3  

    # Post Init
    def __post_init__(self):
        self.SPLIT_DIR.mkdir(exist_ok=True) # If that folder already existed, please don't raise an error.
        self.__validation__()
        

    # Validation (optional)
    def __validation__(self):
        benign_sum = self.BENIGN_TEST_RATIO + self.BENIGN_VAL_RATIO + self.BENIGN_TRAIN_RATIO
        attack_sum = self.ATTACK_TEST_RATIO + self.ATTACK_VAL_RATIO + self.ATTACK_TRAIN_RATIO

        assert abs(benign_sum - 1.0) < 1e-6, f"Benign sum is not 1.0"
        assert abs(attack_sum - 1.0) < 1e-6, f"Attack sum is not 1.0"


class Issue2HighBenignSweepConfig:
    # Purpose:
    # - Stress-test the 3-layer pipeline under highly benign-skewed input mixes.
    # - Run both mode 11 and mode 12 on the exact same sampled subsets so the
    #   comparison between orders stays fair.

    MODES = [11, 12]

    # Repeats are intentionally low because each run evaluates both modes.
    REPEATS = 1

    # Default benign ratio range shown in the menu prompt.
    BENIGN_START_PCT = 90
    BENIGN_END_PCT = 99
    BENIGN_STEP_PCT = 1

    # Max fixed sample size that still fits the current test set under 99%
    # benign when using int(total_size * ratio) for benign selection.
    FIXED_TOTAL_SIZE = 2772

    # Sampling policy:
    # 1) First, try to keep attacks balanced across prompt/rag/tool/correlated.
    # 2) If the exact malicious budget is not divisible or a group is short,
    #    fill the remainder by random sampling from the remaining malicious pool.
    ATTACK_TYPES = ["prompt", "rag", "tool", "correlated"]
    RANDOM_SEED = SEED

    # Output folder under results/
    OUTPUT_SUBDIR = "distribution_sweep_high_benign"
