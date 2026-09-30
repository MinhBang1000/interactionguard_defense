# ============================================================================
# FILE: training/sentinel_modernbert_finetune.py
# Fine-tune an alternative Layer 2 classifier using the Sentinel backbone.
#
# Purpose:
# - Provides an alternate training path for Layer 2 using Sentinel-style
#   backbone choices instead of the default ProtectAI model.
#
# Workflow:
# - Mirrors the standard Layer 2 fine-tuning path.
# - Saves the trained model, tokenizer, and training configuration.
#
# Use this file when:
# - You want to compare Layer 2 backbone choices experimentally.
# ============================================================================
import json
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from config.config import Layer2SentinelConfig
from utils.utils import load_jsonl, make_dataset_for_layer2

os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_DEVICE_ML", "0")


# ============================================================================
# CONFIG
# ============================================================================


config = Layer2SentinelConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# LOAD JSONL
# ============================================================================

train_data = load_jsonl(config.DATA_DIR / config.TRAIN_FILE)
val_data = load_jsonl(config.DATA_DIR / config.VAL_FILE)

if not train_data:
    raise ValueError(f"Empty training file: {config.DATA_DIR / config.TRAIN_FILE}")

if not val_data:
    raise ValueError(f"Empty validation file: {config.DATA_DIR / config.VAL_FILE}")


# ============================================================================
# SUBSET TRAINING DATA
# ============================================================================

np.random.seed(config.SEED)
np.random.shuffle(train_data)

subset_size = int(len(train_data) * config.TRAIN_RATIO)
train_data = train_data[:subset_size]

print(f"Training samples used: {subset_size}")
print(f"Validation samples    : {len(val_data)}")


# ============================================================================
# DATASET
# ============================================================================


train_dataset = make_dataset_for_layer2(train_data)
val_dataset = make_dataset_for_layer2(val_data)


# ============================================================================
# TOKENIZER
# ============================================================================

tokenizer = AutoTokenizer.from_pretrained(
    config.MODEL_NAME,
    trust_remote_code=config.TRUST_REMOTE_CODE,
)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=config.MAX_LENGTH,
    )

train_dataset = train_dataset.map(tokenize, batched=False)
val_dataset = val_dataset.map(tokenize, batched=False)

train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])


# ============================================================================
# MODEL
# ============================================================================

model = AutoModelForSequenceClassification.from_pretrained(
    config.MODEL_NAME,
    num_labels=config.NUM_LABELS,
    trust_remote_code=config.TRUST_REMOTE_CODE,
).to(device)


# ============================================================================
# METRICS
# ============================================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ============================================================================
# RUN
# ============================================================================

def run():
    training_args = TrainingArguments(
        # Output
        output_dir=str(config.OUTPUT_DIR),

        # Training
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,

        # Optimization
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        gradient_accumulation_steps=config.GRAD_ACCUM,

        # Precision
        fp16=config.FP16,

        # Eval / Save
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging
        logging_dir=str(config.OUTPUT_DIR / "logs"),
        logging_steps=config.LOGGING_STEPS,
        report_to="none",

        # Reproducibility
        seed=config.SEED,
        data_seed=config.SEED,

        # DataLoader
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=True,
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.EARLY_STOP_PATIENCE
            )
        ],
    )

    # -------------------------
    # TRAIN
    # -------------------------
    trainer.train()

    # -------------------------
    # VALIDATION EVALUATION
    # -------------------------
    results = trainer.evaluate(eval_dataset=val_dataset)

    print("\nValidation Results")
    for k, v in results.items():
        if "eval" in k:
            try:
                print(f"{k}: {v:.4f}")
            except Exception:
                print(f"{k}: {v}")

    # -------------------------
    # SAVE MODEL
    # -------------------------
    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    # -------------------------
    # SAVE TRAINING CONFIG
    # -------------------------
    training_config = {
        "model_name": config.MODEL_NAME,
        "num_labels": config.NUM_LABELS,
        "max_length": config.MAX_LENGTH,
        "trust_remote_code": config.TRUST_REMOTE_CODE,

        "epochs": config.EPOCHS,
        "learning_rate": config.LR,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation": config.GRAD_ACCUM,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_ratio": config.WARMUP_RATIO,
        "max_grad_norm": config.MAX_GRAD_NORM,

        "train_samples": subset_size,
        "val_samples": len(val_data),
        "output_dir": str(config.OUTPUT_DIR),
    }

    with open(config.OUTPUT_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)

    print("\nTraining config saved:", config.OUTPUT_DIR / "training_config.json")
    print("\nModel saved:", config.OUTPUT_DIR)


if __name__ == "__main__":
    run()
