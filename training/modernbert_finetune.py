# ============================================================================
# FILE: training/modernbert_finetune.py
# Fine-tune the ModernBERT encoder used as the Layer 1 backbone.
#
# Purpose:
# - Trains the encoder that later supplies embeddings to the Layer 1
#   autoencoder-based anomaly detector.
#
# Workflow:
# - Loads processed training and validation data.
# - Fine-tunes ModernBERT with early stopping and evaluation checkpoints.
# - Saves the resulting encoder artifacts and training metadata.
#
# Use this file when:
# - You want to rebuild the encoder stage before retraining Layer 1.
# ============================================================================
# ============================================================================
# Minimal ModernBERT Fine-tuning
# Keeps:
# - EarlyStopping
# - Validation evaluation
# - Train subset control
# ============================================================================

import json
import os
import argparse
import torch
import numpy as np
from utils.utils import load_jsonl, make_dataset
from config.config import ModernBERTConfig
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# ==============================
# METRICS
# ==============================

def compute_metrics(pred):

    logits, labels = pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary"
    )

    acc = accuracy_score(labels, preds)

    return {
        "accuracy":acc,
        "precision":precision,
        "recall":recall,
        "f1":f1
    }

def run(train_percent: int | None = None):
    # ==============================
    # TRAINING ARGUMENTS
    # ==============================
    config = ModernBERTConfig()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA_VISIBLE_DEVICES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_ratio = config.TRAIN_RATIO if train_percent is None else (train_percent / 100.0)
    if not 0 < train_ratio <= 1:
        raise ValueError("train_percent must be between 1 and 100")

    train_data = load_jsonl(config.DATA_DIR / config.TRAIN_FILE)
    val_data = load_jsonl(config.DATA_DIR / config.VAL_FILE)
    if not train_data:
        raise ValueError(f"Empty training file: {config.DATA_DIR / config.TRAIN_FILE}")
    if not val_data:
        raise ValueError(f"Empty validation file: {config.DATA_DIR / config.VAL_FILE}")

    np.random.seed(config.SEED)
    np.random.shuffle(train_data)
    subset_size = max(1, int(len(train_data) * train_ratio))
    train_data = train_data[:subset_size]

    print(f"Training ratio used : {train_ratio:.2%}")
    print(f"Training samples    : {subset_size}")
    print(f"Validation samples  : {len(val_data)}")
    print(f"CUDA devices        : {config.CUDA_VISIBLE_DEVICES}")

    train_dataset = make_dataset(train_data)
    val_dataset = make_dataset(val_data)

    tokenizer = AutoTokenizer.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True
    )

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=config.MAX_LENGTH
        )

    train_dataset = train_dataset.map(tokenize)
    val_dataset = val_dataset.map(tokenize)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=2,
        trust_remote_code=True
    ).to(device)

    training_args = TrainingArguments(

        # Output
        output_dir=str(config.OUTPUT_DIR),

        # Training
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=16,
        learning_rate=config.LR,

        # Optimization (RESTORED)
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        gradient_accumulation_steps=config.GRAD_ACCUM,

        # Mixed precision
        fp16=torch.cuda.is_available(),

        # Evaluation (RESTORED)
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,

        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        # Logging (RESTORED)
        logging_dir=str(config.OUTPUT_DIR / "logs"),
        logging_steps=config.LOGGING_STEPS,
        report_to="none",

        # Reproducibility (RESTORED)
        seed=config.SEED,
        data_seed=config.SEED,

        # DataLoader stability (RESTORED)
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        dataloader_drop_last=False
    )

    # ==============================
    # TRAINER
    # ==============================

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
        ]
    )
    # ==============================
    # TRAIN
    # ==============================

    trainer.train()

    # ==============================
    # VALIDATION EVALUATION
    # ==============================

    results = trainer.evaluate(eval_dataset=val_dataset)

    print("\nValidation Results")

    for k,v in results.items():
        if "eval" in k:
            print(f"{k}: {v:.4f}")

    # ==============================
    # SAVE MODEL
    # ==============================

    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

    # ==============================
    # ADDED: SAVE TRAINING CONFIG
    # ==============================

    training_config = {

        "model_name": config.MODEL_NAME,
        "max_length": config.MAX_LENGTH,
        "trust_remote_code": True,

        "epochs": config.EPOCHS,
        "learning_rate": config.LR,
        "batch_size": config.BATCH_SIZE,
        "gradient_accumulation": config.GRAD_ACCUM,
        "train_ratio": train_ratio,
        "train_samples": subset_size,
        "val_samples": len(val_data),
    }

    with open(config.OUTPUT_DIR / "training_config.json","w") as f:
        json.dump(training_config,f,indent=2)

    print("\nTraining config saved:", config.OUTPUT_DIR / "training_config.json")

    print("\nModel saved:", config.OUTPUT_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Layer 1 ModernBERT encoder.")
    parser.add_argument("--train_percent", type=int, default=None)
    args = parser.parse_args()
    run(train_percent=args.train_percent)
