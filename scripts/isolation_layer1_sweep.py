#!/usr/bin/env python3
"""
Isolation experiment for Layer 1 encoder + autoencoder sweeps.

This script intentionally does not call the project's training entrypoints,
because those entrypoints write to the main model directories. All artifacts
from this experiment are saved under:

    results/isolation_layer1_sweep/<run_id>/

Workflow:
1. Fine-tune ModernBERT for each train percentage.
2. Train a fresh Layer 1 autoencoder on benign embeddings from that encoder.
3. Evaluate full Mode 11 (Layer 1 -> Layer 2 -> Layer 3) thresholds on a
   fixed 99% benign subset.
4. Pick the best option with precision/recall/F1 >= 0.80 when available.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_DEVICE_ML", "0"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from models.detectors.pipeline import DetectionPipeline


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULT_ROOT = PROJECT_ROOT / "results" / "isolation_layer1_sweep"

DEFAULT_THRESHOLDS = ",".join(f"{value / 100:.2f}" for value in range(10, 21))


def stage(message: str) -> None:
    print(f"\n[Isolation Sweep] {message}", flush=True)


@dataclass
class ExperimentConfig:
    model_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 2048
    seed: int = 42
    cuda_visible_devices: str = os.getenv("CUDA_DEVICE_ML", "0")

    encoder_epochs: int = 3
    encoder_lr: float = 2e-5
    encoder_batch_size: int = 4
    encoder_grad_accum: int = 8
    encoder_eval_steps: int = 500
    encoder_logging_steps: int = 100
    encoder_save_total_limit: int = 2
    encoder_early_stop_patience: int = 2
    encoder_weight_decay: float = 0.01
    encoder_warmup_ratio: float = 0.1
    encoder_max_grad_norm: float = 1.0

    ae_embedding_dim: int = 768
    ae_hidden_dims: Tuple[int, ...] = (512, 256, 128)
    ae_latent_dim: int = 64
    ae_dropout: float = 0.1
    ae_epochs: int = 50
    ae_batch_size: int = 128
    ae_lr: float = 1e-3
    ae_weight_decay: float = 1e-5
    ae_patience: int = 10
    ae_min_delta: float = 1e-4

    fixed_total_size: int = 2772
    benign_ratio: float = 0.99


def load_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Dict | List) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, dict):
            return {key: convert(value) for key, value in obj.items()}
        if isinstance(obj, list):
            return [convert(value) for value in obj]
        return obj

    with path.open("w", encoding="utf-8") as f:
        json.dump(convert(data), f, indent=2, ensure_ascii=False)


def make_dataset(rows: List[Dict]) -> Dataset:
    return Dataset.from_dict({
        "text": [row["sequence"] for row in rows],
        "label": [int(row["label"]) for row in rows],
    })


def compute_encoder_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def fine_tune_encoder(
    cfg: ExperimentConfig,
    train_data: List[Dict],
    val_data: List[Dict],
    train_percent: int,
    output_dir: Path,
) -> Path:
    rng = np.random.default_rng(cfg.seed)
    shuffled = list(train_data)
    rng.shuffle(shuffled)

    train_ratio = train_percent / 100.0
    subset_size = max(1, int(len(shuffled) * train_ratio))
    train_subset = shuffled[:subset_size]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)

    train_ds = make_dataset(train_subset)
    val_ds = make_dataset(val_data)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tokenize)
    val_ds = val_ds.map(tokenize)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=2,
        trust_remote_code=True,
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=cfg.encoder_epochs,
        per_device_train_batch_size=cfg.encoder_batch_size,
        per_device_eval_batch_size=16,
        learning_rate=cfg.encoder_lr,
        weight_decay=cfg.encoder_weight_decay,
        warmup_ratio=cfg.encoder_warmup_ratio,
        max_grad_norm=cfg.encoder_max_grad_norm,
        gradient_accumulation_steps=cfg.encoder_grad_accum,
        fp16=torch.cuda.is_available(),
        eval_strategy="steps",
        eval_steps=cfg.encoder_eval_steps,
        save_strategy="steps",
        save_steps=cfg.encoder_eval_steps,
        save_total_limit=cfg.encoder_save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(output_dir / "logs"),
        logging_steps=cfg.encoder_logging_steps,
        report_to="none",
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
        dataloader_pin_memory=torch.cuda.is_available(),
        dataloader_drop_last=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_encoder_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=cfg.encoder_early_stop_patience
            )
        ],
    )

    trainer.train()
    eval_results = trainer.evaluate(eval_dataset=val_ds)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    save_json(output_dir / "training_config.json", {
        "model_name": cfg.model_name,
        "max_length": cfg.max_length,
        "trust_remote_code": True,
        "epochs": cfg.encoder_epochs,
        "learning_rate": cfg.encoder_lr,
        "batch_size": cfg.encoder_batch_size,
        "gradient_accumulation": cfg.encoder_grad_accum,
        "train_ratio": train_ratio,
        "train_percent": train_percent,
        "train_samples": subset_size,
        "val_samples": len(val_data),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "eval_results": eval_results,
        "isolation_output_dir": str(output_dir),
    })

    return output_dir


class EmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, values: np.ndarray):
        self.values = torch.FloatTensor(values)

    def __len__(self) -> int:
        return len(self.values)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.values[index]


class AutoEncoder(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        encoder_layers = []
        in_dim = cfg.ae_embedding_dim
        for hidden_dim in cfg.ae_hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.ae_dropout),
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, cfg.ae_latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        decoder_layers = []
        in_dim = cfg.ae_latent_dim
        for hidden_dim in reversed(cfg.ae_hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.ae_dropout),
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, cfg.ae_embedding_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, values: torch.Tensor):
        latent = self.encoder(values)
        reconstructed = self.decoder(latent)
        return reconstructed, latent


def extract_embeddings(
    encoder_dir: Path,
    rows: List[Dict],
    cfg: ExperimentConfig,
    device: torch.device,
    batch_size: int = 32,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, trust_remote_code=True)
    encoder = AutoModel.from_pretrained(encoder_dir, trust_remote_code=True).to(device)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    embeddings = []
    with torch.no_grad():
        for start in tqdm(range(0, len(rows), batch_size), desc="Extract embeddings", unit="batch"):
            batch = rows[start:start + batch_size]
            texts = [row["sequence"] for row in batch]
            inputs = tokenizer(
                texts,
                max_length=cfg.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            outputs = encoder(**inputs)
            cls = outputs.last_hidden_state[:, 0, :]
            embeddings.append(cls.cpu().numpy())

    del encoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.vstack(embeddings)


def train_autoencoder(
    cfg: ExperimentConfig,
    train_embeddings: np.ndarray,
    val_embeddings: np.ndarray,
    output_path: Path,
    device: torch.device,
) -> Path:
    train_loader = DataLoader(
        EmbeddingDataset(train_embeddings),
        batch_size=cfg.ae_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        EmbeddingDataset(val_embeddings),
        batch_size=cfg.ae_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = AutoEncoder(cfg).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.ae_lr,
        weight_decay=cfg.ae_weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    best_val = float("inf")
    best_state = None
    patience = 0
    history = {"train": [], "val": []}

    epoch_bar = tqdm(range(cfg.ae_epochs), desc="Train Layer 1 AutoEncoder", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            reconstructed, _ = model(batch)
            loss = criterion(reconstructed, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                reconstructed, _ = model(batch)
                loss = criterion(reconstructed, batch)
                val_loss += loss.item()
        val_loss /= max(1, len(val_loader))

        scheduler.step(val_loss)
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        epoch_bar.set_postfix({
            "train": f"{train_loss:.6f}",
            "val": f"{val_loss:.6f}",
            "best": f"{best_val:.6f}" if math.isfinite(best_val) else "inf",
        })

        if val_loss < best_val - cfg.ae_min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= cfg.ae_patience:
                print("AE early stopping")
                break

    if best_state:
        model.load_state_dict(best_state)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "input_dim": cfg.ae_embedding_dim,
                "hidden_dims": list(cfg.ae_hidden_dims),
                "latent_dim": cfg.ae_latent_dim,
                "dropout": cfg.ae_dropout,
            },
            "history": history,
            "best_val_loss": best_val,
        },
        output_path,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return output_path


def compute_reconstruction_errors(
    cfg: ExperimentConfig,
    ae_path: Path,
    embeddings: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    checkpoint = torch.load(ae_path, map_location=device)
    model = AutoEncoder(cfg).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    errors = []
    with torch.no_grad():
        for start in tqdm(range(0, len(embeddings), cfg.ae_batch_size), desc="AE errors", unit="batch"):
            batch = torch.FloatTensor(embeddings[start:start + cfg.ae_batch_size]).to(device)
            reconstructed, _ = model(batch)
            mse = torch.mean((batch - reconstructed) ** 2, dim=1)
            errors.append(mse.cpu().numpy())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return np.concatenate(errors)


def sample_high_benign_99(rows: List[Dict], cfg: ExperimentConfig) -> List[Dict]:
    rng = random.Random(cfg.seed + 9900)
    benign = [row for row in rows if int(row["label"]) == 0]
    malicious_by_type = {
        attack_type: [
            row for row in rows
            if int(row["label"]) == 1 and row.get("attack_type") == attack_type
        ]
        for attack_type in ["prompt", "rag", "tool", "correlated"]
    }

    n_benign = int(cfg.fixed_total_size * cfg.benign_ratio)
    n_malicious = cfg.fixed_total_size - n_benign
    if n_benign > len(benign):
        raise ValueError(f"Need {n_benign} benign rows but only found {len(benign)}")

    selected = rng.sample(benign, n_benign)
    per_type = n_malicious // len(malicious_by_type)
    remaining_needed = n_malicious
    remaining_pool = []

    for pool in malicious_by_type.values():
        take = min(per_type, len(pool), remaining_needed)
        chosen = rng.sample(pool, take)
        selected.extend(chosen)
        remaining_needed -= take
        chosen_ids = {id(row) for row in chosen}
        remaining_pool.extend([row for row in pool if id(row) not in chosen_ids])

    if remaining_needed > 0:
        selected.extend(rng.sample(remaining_pool, remaining_needed))

    rng.shuffle(selected)
    return selected


def evaluate_thresholds(
    labels: np.ndarray,
    errors: np.ndarray,
    thresholds: Iterable[float],
) -> List[Dict]:
    results = []
    for threshold in tqdm(list(thresholds), desc="Evaluate Layer 1 thresholds", unit="threshold"):
        preds = (errors > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
        )
        tn = int(((labels == 0) & (preds == 0)).sum())
        fp = int(((labels == 0) & (preds == 1)).sum())
        fn = int(((labels == 1) & (preds == 0)).sum())
        tp = int(((labels == 1) & (preds == 1)).sum())
        results.append({
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(labels, preds)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "fpr": float(fp / (fp + tn)) if (fp + tn) else 0.0,
            "fnr": float(fn / (fn + tp)) if (fn + tp) else 0.0,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "num_detected": int(preds.sum()),
        })
    return results


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_mode11_config(
    output_path: Path,
    encoder_dir: Path,
    ae_path: Path,
    threshold: float,
) -> Path:
    with (PROJECT_ROOT / "config" / "detection_config.yaml").open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    # CUDA_VISIBLE_DEVICES exposes the selected physical GPU as logical cuda:0
    # inside this process. Use the default visible CUDA device in the temporary
    # config instead of copying a physical cuda index from the runtime config.
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    config["test_mode"] = 11
    layer1_params = config.setdefault("layers", {}).setdefault("layer1", {}).setdefault("params", {})
    layer1_params["bert_model_path"] = str(encoder_dir)
    layer1_params["ae_model_path"] = str(ae_path)
    layer1_params["threshold"] = float(threshold)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)
    return output_path


def attach_isolated_layer1_cache(pipeline: DetectionPipeline, cache_dir: Path) -> None:
    layer1 = pipeline.layers.get("layer1")
    if layer1 is None:
        return

    original_extract = layer1.extract_embedding

    def extract_with_isolated_cache(sequences, batch_size=32, cache_dir_override=None):
        return original_extract(
            sequences,
            batch_size=batch_size,
            cache_dir=str(cache_dir),
        )

    layer1.extract_embedding = extract_with_isolated_cache


def summarize_mode11_result(
    result: Dict,
    threshold: float,
    train_percent: int,
    train_subset_size: int,
    cfg: ExperimentConfig,
    device: torch.device,
) -> Dict:
    metrics = result.get("metrics", {}) or {}
    cm = metrics.get("confusion_matrix", {}) or {}
    layer_stats = result.get("layer_stats", {}) or {}
    layer3_processed = int((layer_stats.get("layer3", {}) or {}).get("processed", 0) or 0)
    num_samples = int(result.get("num_samples", 0) or 0)
    elapsed = float(result.get("time", 0.0) or 0.0)

    return {
        "train_percent": train_percent,
        "train_ratio": train_percent / 100.0,
        "train_samples": train_subset_size,
        "threshold": float(threshold),
        "test_mode": 11,
        "model_name": cfg.model_name,
        "max_length": cfg.max_length,
        "encoder_epochs": cfg.encoder_epochs,
        "encoder_lr": cfg.encoder_lr,
        "encoder_batch_size": cfg.encoder_batch_size,
        "encoder_grad_accum": cfg.encoder_grad_accum,
        "ae_epochs": cfg.ae_epochs,
        "ae_lr": cfg.ae_lr,
        "ae_batch_size": cfg.ae_batch_size,
        "ae_latent_dim": cfg.ae_latent_dim,
        "benign_ratio": cfg.benign_ratio,
        "fixed_total_size": cfg.fixed_total_size,
        "device": str(device),
        "num_samples": num_samples,
        "num_detected": int(result.get("num_detected", 0) or 0),
        "time": elapsed,
        "latency_per_sample": elapsed / num_samples if num_samples else 0.0,
        "layer1_processed": int((layer_stats.get("layer1", {}) or {}).get("processed", 0) or 0),
        "layer1_sent_to_next": int((layer_stats.get("layer1", {}) or {}).get("sent_to_next", 0) or 0),
        "layer2_processed": int((layer_stats.get("layer2", {}) or {}).get("processed", 0) or 0),
        "layer2_blocked": int((layer_stats.get("layer2", {}) or {}).get("blocked", 0) or 0),
        "layer2_sent_to_next": int((layer_stats.get("layer2", {}) or {}).get("sent_to_next", 0) or 0),
        "layer3_processed": layer3_processed,
        "layer3_processing_ratio": layer3_processed / num_samples if num_samples else 0.0,
        "layer3_blocked": int((layer_stats.get("layer3", {}) or {}).get("blocked", 0) or 0),
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("f1_score", 0.0) or 0.0),
        "f1_score": float(metrics.get("f1_score", 0.0) or 0.0),
        "fpr": float(metrics.get("fpr", 0.0) or 0.0),
        "fnr": float(metrics.get("fnr", 0.0) or 0.0),
        "tp": int(cm.get("tp", 0) or 0),
        "fp": int(cm.get("fp", 0) or 0),
        "tn": int(cm.get("tn", 0) or 0),
        "fn": int(cm.get("fn", 0) or 0),
    }


def choose_best(results: List[Dict]) -> Dict:
    eligible = [
        row for row in results
        if row["precision"] >= 0.80 and row["recall"] >= 0.80 and row.get("f1", row.get("f1_score", 0.0)) >= 0.80
    ]
    pool = eligible if eligible else results
    best = max(
        pool,
        key=lambda row: (
            row.get("f1", row.get("f1_score", 0.0)),
            row["precision"],
            row["recall"],
            row["accuracy"],
            -row["fpr"],
        ),
    )
    best = dict(best)
    best["meets_80pct_precision_recall_f1"] = bool(eligible)
    return best


def parse_thresholds(raw: str) -> List[float]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            values.append(float(chunk))
    return values


def parse_percents(raw: str) -> List[int]:
    values = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if chunk:
            value = int(chunk)
            if not 1 <= value <= 100:
                raise ValueError("train percents must be between 1 and 100")
            values.append(value)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Layer 1 retrain/sweep experiment.")
    parser.add_argument("--train_percents", default="50,75,100")
    parser.add_argument("--thresholds", default=DEFAULT_THRESHOLDS)
    parser.add_argument("--run_id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--ae_epochs", type=int, default=None)
    parser.add_argument("--encoder_epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = ExperimentConfig()
    if args.ae_epochs is not None:
        cfg.ae_epochs = args.ae_epochs
    if args.encoder_epochs is not None:
        cfg.encoder_epochs = args.encoder_epochs

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.cuda_visible_devices
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = RESULT_ROOT / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    train_percents = parse_percents(args.train_percents)
    thresholds = parse_thresholds(args.thresholds)

    stage(f"Run directory: {run_dir}")
    stage(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    stage(f"Train percentages: {train_percents}")
    stage(f"Thresholds: {thresholds}")

    save_json(run_dir / "experiment_config.json", {
        "config": asdict(cfg),
        "train_percents": train_percents,
        "thresholds": thresholds,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    })

    train_data = load_jsonl(PROCESSED_DIR / "train_processed.jsonl")
    val_data = load_jsonl(PROCESSED_DIR / "val_processed.jsonl")
    test_data = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")

    train_benign = [row for row in train_data if int(row["label"]) == 0]
    val_benign = [row for row in val_data if int(row["label"]) == 0]
    high_benign_eval = sample_high_benign_99(test_data, cfg)
    eval_labels = np.array([int(row["label"]) for row in high_benign_eval])

    all_results = []

    for train_percent in tqdm(train_percents, desc="Layer 1 isolation sweep", unit="config"):
        stage(f"Train percent {train_percent}%")

        percent_dir = run_dir / f"train_pct_{train_percent:03d}"
        encoder_dir = percent_dir / "modernbert_encoder"
        ae_path = percent_dir / "autoencoder.pth"
        train_subset_size = max(1, int(len(train_data) * (train_percent / 100.0)))

        stage("Fine-tuning ModernBERT encoder")
        fine_tune_encoder(cfg, train_data, val_data, train_percent, encoder_dir)

        stage("Extracting benign train embeddings")
        train_embeddings = extract_embeddings(encoder_dir, train_benign, cfg, device)
        stage("Extracting benign validation embeddings")
        val_embeddings = extract_embeddings(encoder_dir, val_benign, cfg, device)
        stage("Training fresh Layer 1 AutoEncoder")
        train_autoencoder(cfg, train_embeddings, val_embeddings, ae_path, device)

        stage("Sweeping full Mode 11 thresholds on high-benign 99% evaluation set")
        sequences = [row["sequence"] for row in high_benign_eval]
        threshold_results = []
        pipeline = None

        for threshold in tqdm(thresholds, desc="Evaluate Mode 11 thresholds", unit="threshold"):
            threshold_tag = f"{threshold:.2f}".replace(".", "_")
            config_path = write_mode11_config(
                percent_dir / "mode11_configs" / f"threshold_{threshold_tag}.yaml",
                encoder_dir=encoder_dir,
                ae_path=ae_path,
                threshold=threshold,
            )

            if pipeline is None:
                pipeline = DetectionPipeline(config_path=str(config_path), test_mode=11)
                attach_isolated_layer1_cache(pipeline, percent_dir / "embedding_cache")
            else:
                pipeline.layers["layer1"].threshold = float(threshold)

            result = pipeline.predict(sequences, labels=eval_labels, return_details=False)
            result["isolation_metadata"] = {
                "train_percent": train_percent,
                "train_samples": train_subset_size,
                "threshold": float(threshold),
                "encoder_dir": str(encoder_dir),
                "autoencoder_path": str(ae_path),
                "mode11_config_path": str(config_path),
                "benign_ratio": cfg.benign_ratio,
                "fixed_total_size": cfg.fixed_total_size,
            }
            save_json(percent_dir / "mode11_results" / f"threshold_{threshold_tag}.json", result)

            threshold_results.append(
                summarize_mode11_result(
                    result=result,
                    threshold=threshold,
                    train_percent=train_percent,
                    train_subset_size=train_subset_size,
                    cfg=cfg,
                    device=device,
                )
            )

        best = choose_best(threshold_results)
        save_json(percent_dir / "mode11_threshold_results.json", threshold_results)
        save_json(percent_dir / "best_result.json", best)
        save_csv(percent_dir / "mode11_threshold_results.csv", threshold_results)

        all_results.extend(threshold_results)
        stage(f"Best for {train_percent}%: {best}")
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    global_best = choose_best(all_results)
    save_json(run_dir / "all_threshold_results.json", all_results)
    save_json(run_dir / "candidate_results.json", all_results)
    save_json(run_dir / "best_overall.json", global_best)
    save_csv(run_dir / "candidate_results.csv", all_results)

    print("\n" + "=" * 80)
    print("ISOLATION LAYER 1 SWEEP COMPLETE")
    print("=" * 80)
    print(f"Run dir      : {run_dir}")
    print(f"Best overall : {global_best}")


if __name__ == "__main__":
    main()
