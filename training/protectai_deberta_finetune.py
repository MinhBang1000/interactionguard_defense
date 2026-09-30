# ============================================================================
# FILE: training/protectai_deberta_finetune.py
# Fine-tune the default Layer 2 ProtectAI DeBERTa classifier.
#
# Purpose:
# - Trains the primary supervised classifier used in Layer 2.
#
# Workflow:
# - Loads processed train/validation data.
# - Excludes correlated attacks so Layer 2 focuses on known attack families.
# - Applies a configurable train subset ratio.
# - Tokenizes sequences for the ProtectAI model and saves the resulting model.
#
# Use this file when:
# - You want to rebuild the default Layer 2 model with a controlled amount of
#   data while keeping correlated attacks for Layer 3.
# ============================================================================
import json
import os
import argparse

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

from config.config import Layer2ProtectAIConfig
from utils.utils import load_jsonl, make_dataset_for_layer2


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


def run(train_percent: int | None = None):
    config = Layer2ProtectAIConfig()
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

    train_data = [item for item in train_data if item.get("attack_type") != "correlated"]
    val_data = [item for item in val_data if item.get("attack_type") != "correlated"]

    np.random.seed(config.SEED)
    np.random.shuffle(train_data)

    subset_size = max(1, int(len(train_data) * train_ratio))
    train_data = train_data[:subset_size]

    print(f"Training ratio used        : {train_ratio:.2%}")
    print(f"Training samples used      : {subset_size}")
    print(f"Validation samples used    : {len(val_data)}")
    print(f"CUDA devices              : {config.CUDA_VISIBLE_DEVICES}")
    print("Excluded attack types      : correlated")

    train_dataset = make_dataset_for_layer2(train_data)
    val_dataset = make_dataset_for_layer2(val_data)

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

    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME,
        num_labels=config.NUM_LABELS,
        trust_remote_code=config.TRUST_REMOTE_CODE,
    ).to(device)

    training_args = TrainingArguments(
        output_dir=str(config.OUTPUT_DIR),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        fp16=config.FP16,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.EVAL_STEPS,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(config.OUTPUT_DIR / "logs"),
        logging_steps=config.LOGGING_STEPS,
        report_to="none",
        seed=config.SEED,
        data_seed=config.SEED,
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

    trainer.train()

    results = trainer.evaluate(eval_dataset=val_dataset)

    print("\nValidation Results")
    for k, v in results.items():
        if "eval" in k:
            try:
                print(f"{k}: {v:.4f}")
            except Exception:
                print(f"{k}: {v}")

    trainer.save_model(config.OUTPUT_DIR)
    tokenizer.save_pretrained(config.OUTPUT_DIR)

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
        "train_ratio": train_ratio,
        "excluded_attack_types": ["correlated"],
        "train_samples": subset_size,
        "val_samples": len(val_data),
        "output_dir": str(config.OUTPUT_DIR),
    }

    with open(config.OUTPUT_DIR / "training_config.json", "w", encoding="utf-8") as f:
        json.dump(training_config, f, indent=2, ensure_ascii=False)

    print("\nTraining config saved:", config.OUTPUT_DIR / "training_config.json")
    print("\nModel saved:", config.OUTPUT_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Layer 2 ProtectAI model.")
    parser.add_argument("--train_percent", type=int, default=None)
    args = parser.parse_args()
    run(train_percent=args.train_percent)
