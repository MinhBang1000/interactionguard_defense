#!/usr/bin/env python3
"""
Re-evaluate previously trained isolation Layer 1 models with the current pipeline.

This script does not fine-tune ModernBERT and does not retrain the AutoEncoder.
It reuses artifacts from:

    results/isolation_layer1_sweep/<run_id>/train_pct_XXX/

and reruns full Mode 11 (Layer 1 -> Layer 2 -> Layer 3), which is useful for
Layer 3 prompt ablations and FP comparisons.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_DEVICE_ML", "0"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from tqdm.auto import tqdm

from models.detectors.pipeline import DetectionPipeline
from scripts.isolation_layer1_sweep import (
    ExperimentConfig,
    attach_isolated_layer1_cache,
    load_jsonl,
    sample_high_benign_99,
    save_json,
    summarize_mode11_result,
    write_mode11_config,
)


RESULT_ROOT = PROJECT_ROOT / "results" / "isolation_layer1_sweep"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def stage(message: str) -> None:
    print(f"\n[Isolation Re-eval] {message}", flush=True)


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def latest_completed_run(train_percent: int) -> str:
    required = Path(f"train_pct_{train_percent:03d}") / "best_result.json"
    runs = [
        path for path in RESULT_ROOT.iterdir()
        if path.is_dir()
        and path.name[:8].isdigit()
        and (path / "experiment_config.json").exists()
        and (path / required).exists()
    ]
    if not runs:
        raise FileNotFoundError(
            f"No completed isolation run for train_percent={train_percent} under {RESULT_ROOT}"
        )
    return sorted(runs, key=lambda path: path.name)[-1].name


def parse_percents(raw: str) -> List[int]:
    return [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def default_thresholds_from_run(run_dir: Path) -> List[float]:
    config_path = run_dir / "experiment_config.json"
    if config_path.exists():
        thresholds = (load_json(config_path).get("thresholds") or [])
        if thresholds:
            return [float(value) for value in thresholds]
    return [value / 100 for value in range(10, 21)]


def thresholds_for_percent(percent_dir: Path, run_dir: Path, raw: str) -> List[float]:
    raw = raw.strip().lower()
    if raw == "best":
        return [float(load_json(percent_dir / "best_result.json")["threshold"])]
    if raw == "all":
        return default_thresholds_from_run(run_dir)
    return [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]


def threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


def extract_false_positives(
    eval_rows: List[Dict],
    predictions: List[int],
    scores: List[float],
    run_id: str,
    train_percent: int,
    threshold: float,
    result_path: Path,
) -> List[Dict]:
    false_positives = []
    for index, (row, pred, score) in enumerate(zip(eval_rows, predictions, scores)):
        label = int(row["label"])
        if label == 0 and int(pred) == 1:
            fp_row = dict(row)
            fp_row.update({
                "eval_index": index,
                "predicted_label": int(pred),
                "true_label": label,
                "pipeline_score": float(score),
                "run_id": run_id,
                "train_percent": train_percent,
                "threshold": float(threshold),
                "source_result_path": str(result_path),
            })
            false_positives.append(fp_row)
    return false_positives


def load_experiment_config(run_dir: Path) -> ExperimentConfig:
    cfg = ExperimentConfig()
    config_path = run_dir / "experiment_config.json"
    if not config_path.exists():
        return cfg
    raw = load_json(config_path)
    for key, value in (raw.get("config") or {}).items():
        if hasattr(cfg, key):
            setattr(cfg, key, value)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate trained isolation models with current Mode 11 pipeline.")
    parser.add_argument("--run_id", default=None, help="Isolation run id. Defaults to latest completed run.")
    parser.add_argument("--train_percents", default="50", help="Comma-separated percents, e.g. 50,75,100.")
    parser.add_argument(
        "--thresholds",
        default="best",
        help="best, all, or comma-separated thresholds such as 0.18,0.19.",
    )
    parser.add_argument("--output_name", default=None, help="Output folder name under the run directory.")
    args = parser.parse_args()

    train_percents = parse_percents(args.train_percents)
    run_id = args.run_id or latest_completed_run(train_percents[0])
    run_dir = RESULT_ROOT / run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    output_name = args.output_name or f"reeval_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = run_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_experiment_config(run_dir)
    test_data = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")
    eval_rows = sample_high_benign_99(test_data, cfg)
    eval_labels = np.array([int(row["label"]) for row in eval_rows])
    sequences = [row["sequence"] for row in eval_rows]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    stage(f"Run directory: {run_dir}")
    stage(f"Output directory: {output_dir}")
    stage(f"Device: {device} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    stage(f"Train percents: {train_percents}")
    stage(f"Threshold mode: {args.thresholds}")

    save_json(output_dir / "reeval_config.json", {
        "run_id": run_id,
        "train_percents": train_percents,
        "thresholds": args.thresholds,
        "device": str(device),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "num_eval_samples": len(eval_rows),
    })

    all_rows = []

    for train_percent in tqdm(train_percents, desc="Re-evaluate trained configs", unit="config"):
        percent_dir = run_dir / f"train_pct_{train_percent:03d}"
        encoder_dir = percent_dir / "modernbert_encoder"
        ae_path = percent_dir / "autoencoder.pth"
        if not encoder_dir.exists() or not ae_path.exists():
            raise FileNotFoundError(f"Missing trained artifacts under {percent_dir}")

        thresholds = thresholds_for_percent(percent_dir, run_dir, args.thresholds)
        stage(f"Train percent {train_percent}% | thresholds={thresholds}")

        pipeline = None
        percent_output = output_dir / f"train_pct_{train_percent:03d}"
        train_samples = int(load_json(percent_dir / "best_result.json").get("train_samples", 0))

        for threshold in tqdm(thresholds, desc=f"Mode 11 re-eval {train_percent}%", unit="threshold"):
            tag = threshold_tag(threshold)
            config_path = write_mode11_config(
                percent_output / "mode11_configs" / f"threshold_{tag}.yaml",
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
            result["reeval_metadata"] = {
                "source_run_id": run_id,
                "source_train_percent": train_percent,
                "threshold": float(threshold),
                "encoder_dir": str(encoder_dir),
                "autoencoder_path": str(ae_path),
                "mode11_config_path": str(config_path),
                "purpose": "Layer 3 prompt ablation re-evaluation",
            }

            result_path = percent_output / "mode11_results" / f"threshold_{tag}.json"
            save_json(result_path, result)

            row = summarize_mode11_result(
                result=result,
                threshold=threshold,
                train_percent=train_percent,
                train_subset_size=train_samples,
                cfg=cfg,
                device=device,
            )
            original_best_path = percent_dir / "best_result.json"
            if original_best_path.exists():
                original_best = load_json(original_best_path)
                if abs(float(original_best.get("threshold", -1)) - float(threshold)) < 1e-9:
                    row.update({
                        "original_fp": original_best.get("fp"),
                        "original_fn": original_best.get("fn"),
                        "original_precision": original_best.get("precision"),
                        "original_recall": original_best.get("recall"),
                        "original_f1": original_best.get("f1"),
                        "fp_delta_vs_original_best": row["fp"] - int(original_best.get("fp", 0)),
                    })

            predictions = [int(value) for value in result["predictions"]]
            scores = [float(value) for value in result["probabilities"]]
            fps = extract_false_positives(
                eval_rows=eval_rows,
                predictions=predictions,
                scores=scores,
                run_id=run_id,
                train_percent=train_percent,
                threshold=threshold,
                result_path=result_path,
            )
            fp_path = percent_output / "fp_samples" / f"threshold_{tag}.jsonl"
            save_jsonl(fp_path, fps)
            row["fp_samples_path"] = str(fp_path)

            save_json(percent_output / "summaries" / f"threshold_{tag}.json", row)
            all_rows.append(row)
            stage(
                f"{train_percent}% th={threshold:.2f}: "
                f"FP={row['fp']} FN={row['fn']} P={row['precision']:.4f} "
                f"R={row['recall']:.4f} F1={row['f1']:.4f}"
            )

        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_json(output_dir / "reeval_results.json", all_rows)
    save_csv(output_dir / "reeval_results.csv", all_rows)

    print("\nRe-evaluation complete")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
