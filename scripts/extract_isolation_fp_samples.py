#!/usr/bin/env python3
"""
Extract false-positive samples from an isolation Mode 11 sweep result.

The isolation sweep evaluates a fixed high-benign subset generated from
data/processed/test_processed.jsonl. This script reconstructs that same subset,
aligns it with saved Mode 11 predictions, and writes benign samples that were
blocked by the pipeline to JSONL for manual inspection.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RESULT_ROOT = PROJECT_ROOT / "results" / "isolation_layer1_sweep"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class ExperimentConfig:
    seed: int = 42
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


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def latest_run_id(train_percent: int) -> str:
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
            f"No completed isolation runs for train_percent={train_percent} found under {RESULT_ROOT}"
        )
    return sorted(runs, key=lambda path: path.name)[-1].name


def threshold_tag(threshold: float) -> str:
    return f"{threshold:.2f}".replace(".", "_")


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


def resolve_threshold(percent_dir: Path, requested: str | None) -> float:
    if requested:
        return float(requested)

    best_path = percent_dir / "best_result.json"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Missing {best_path}. Pass --threshold explicitly or finish the sweep first."
        )
    return float(load_json(best_path)["threshold"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract false positives from an isolation Mode 11 result.")
    parser.add_argument("--run_id", default=None, help="Isolation run id. Defaults to latest run.")
    parser.add_argument("--train_percent", type=int, default=50)
    parser.add_argument("--threshold", default=None, help="Layer 1 threshold. Defaults to train_pct best_result.")
    parser.add_argument("--output", default=None, help="Output JSONL path. Defaults inside train_pct directory.")
    args = parser.parse_args()

    run_id = args.run_id or latest_run_id(args.train_percent)
    run_dir = RESULT_ROOT / run_id
    percent_dir = run_dir / f"train_pct_{args.train_percent:03d}"
    threshold = resolve_threshold(percent_dir, args.threshold)
    tag = threshold_tag(threshold)
    result_path = percent_dir / "mode11_results" / f"threshold_{tag}.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Mode 11 result not found: {result_path}")

    cfg = load_experiment_config(run_dir)
    test_data = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")
    eval_rows = sample_high_benign_99(test_data, cfg)
    result = load_json(result_path)
    predictions = [int(value) for value in result["predictions"]]
    scores = result.get("probabilities") or [None] * len(predictions)

    if len(eval_rows) != len(predictions):
        raise ValueError(
            f"Length mismatch: eval_rows={len(eval_rows)} predictions={len(predictions)}"
        )

    false_positives = []
    for index, (row, pred, score) in enumerate(zip(eval_rows, predictions, scores)):
        label = int(row["label"])
        if label == 0 and pred == 1:
            fp_row = dict(row)
            fp_row.update({
                "eval_index": index,
                "predicted_label": pred,
                "true_label": label,
                "pipeline_score": score,
                "run_id": run_id,
                "train_percent": args.train_percent,
                "threshold": threshold,
                "source_result_path": str(result_path),
            })
            false_positives.append(fp_row)

    output_path = Path(args.output) if args.output else percent_dir / f"fp_samples_threshold_{tag}.jsonl"
    summary_path = output_path.with_suffix(".summary.json")

    save_jsonl(output_path, false_positives)
    save_json(summary_path, {
        "run_id": run_id,
        "train_percent": args.train_percent,
        "threshold": threshold,
        "source_result_path": str(result_path),
        "output_path": str(output_path),
        "num_samples": len(eval_rows),
        "num_predictions": len(predictions),
        "num_false_positives": len(false_positives),
        "confusion_matrix": (result.get("metrics") or {}).get("confusion_matrix"),
        "metrics": {
            key: (result.get("metrics") or {}).get(key)
            for key in ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr"]
        },
    })

    print("\nFalse-positive extraction complete")
    print(f"Run           : {run_id}")
    print(f"Train percent : {args.train_percent}")
    print(f"Threshold     : {threshold:.2f}")
    print(f"FP samples    : {len(false_positives)}")
    print(f"JSONL         : {output_path}")
    print(f"Summary       : {summary_path}")


if __name__ == "__main__":
    main()
