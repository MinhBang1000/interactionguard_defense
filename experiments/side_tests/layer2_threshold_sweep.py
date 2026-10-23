"""Validation-only L2 operating-point selection and exploratory test sweep.

No training, no L3 calls, no promotion and no main-menu/config changes.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

GRID = [.5, .6, .7, .8, .85, .9, .95, .97, .98, .99, .995, .999, .9999, 1.0]
BUDGETS = [.01, .005, .001]


def select_fpr(labels, probabilities, budget):
    """Maximize validation TP under an empirical FP budget, without splitting ties."""
    from training.layer1_common import metrics
    labels, probabilities = np.asarray(labels), np.asarray(probabilities, dtype=float)
    if len(labels) != len(probabilities) or set(labels) != {0, 1}:
        raise ValueError("FPR calibration requires aligned validation scores of both classes")
    if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)) or not 0 <= budget <= 1:
        raise ValueError("Invalid probability or FPR budget")
    negatives = int((labels == 0).sum())
    allowed_fp = int(np.floor(negatives * budget + 1e-12))
    candidates = np.unique(np.r_[0., probabilities, 1.])
    feasible = []
    for threshold in candidates:
        m = metrics(labels, probabilities, float(threshold))
        if m["fp"] <= allowed_fp:
            feasible.append((m["tp"], -m["fp"], float(threshold), m))
    chosen = max(feasible, key=lambda item: item[:3])
    return {"threshold": chosen[2], "validation": chosen[3], "fpr_budget": budget,
            "allowed_false_positives": allowed_fp, "validation_negative_samples": negatives,
            "objective": "max_validation_recall_under_empirical_FPR_budget_then_min_FP",
            "nonempty_true_positive_coverage": chosen[0] > 0,
            "comparison": "strict_probability_greater_than_threshold"}


def evaluate_point(labels, errors, probabilities, l1_threshold, l2_threshold):
    from training.layer1_common import metrics
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    eligible = np.ones(len(labels), dtype=bool) if l1_threshold is None else np.asarray(errors) > l1_threshold
    blocked = eligible & (probabilities > l2_threshold)
    pending = eligible & ~blocked
    m = metrics(labels, blocked.astype(float), .5)
    subset = metrics(labels[eligible], probabilities[eligible], l2_threshold)
    return {"scope": "l2_standalone" if l1_threshold is None else "l1_l2_partial",
            "l1_threshold": l1_threshold, "l2_threshold": float(l2_threshold), **m,
            "precision_defined": bool(m["tp"] + m["fp"]),
            "l2_processed": int(eligible.sum()), "l2_blocked": int(blocked.sum()),
            "l3_pending": int(pending.sum()), "l3_pending_ratio": float(pending.mean()),
            "l3_pending_positive": int((pending & (labels == 1)).sum()),
            "l3_pending_negative": int((pending & (labels == 0)).sum()),
            "l1_bypassed_positive": int((~eligible & (labels == 1)).sum()),
            "conditional_l2_precision": subset["precision"], "conditional_l2_recall": subset["recall"],
            "conditional_l2_fpr": subset["fpr"]}


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", choices=["0", "cpu"])
    parser.add_argument("--source", type=Path, default=Path("results/layer2_channelwise/20261021_071550_378314"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
    from data.prepare_training import load_prepared
    from training.layer1_common import directory_hash
    from training.layer1_pipeline import gpu_preflight
    if args.device != "cpu":
        gpu_preflight(args.device)
    source = json.loads((args.source / "config.json").read_text())
    cfg = source["model"]
    if cfg["model_path"] != "protectai/deberta-v3-base-prompt-injection-v2" or not cfg.get("channel_wise"):
        raise ValueError("Expected original channel-wise HF ProtectAI v2")
    for file, expected in source["source_sha256"].items():
        if file.startswith("models/") and sha256(ROOT / file) != expected:
            raise ValueError("Detector source differs from cached test scores")
    if directory_hash(source["snapshot"]) != source["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from cached test scores")
    val_path = ROOT / "data/training/encoder/val.jsonl"
    validation = load_prepared("encoder", "val")
    if sha256(val_path) != source["validation_sha256"]:
        raise ValueError("Validation changed")
    l1_run = Path(source["l1_run_dir"])
    selected = json.loads((l1_run / "autoencoder/selection_frozen.json").read_text())
    val_error_path = Path(selected["best"]["checkpoint"]).parent / "validation_errors.npy"
    if sha256(val_error_path) != source["validation_errors_sha256"]:
        raise ValueError("Validation L1 errors changed")
    val_errors = np.load(val_error_path)
    if len(validation) != len(val_errors):
        raise ValueError("Validation score count mismatch")
    output = args.output_dir or ROOT / "results/side_layer2_thresholds" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output.mkdir(parents=True, exist_ok=False)
    run_config = {"protocol": "validation_FPR_calibration_then_exploratory_inspected_test",
                  "source_results": str(args.source.resolve()), "model": cfg,
                  "checkpoint_sha256": source["checkpoint_sha256"], "l1_thresholds": source["thresholds"],
                  "fixed_L2_grid": GRID, "validation_FPR_budgets": BUDGETS,
                  "validation_scope": "prepared_noncorrelated_no_annotation_review_3514_prefixes",
                  "validation_sha256": sha256(val_path), "physical_gpu": args.device,
                  "source_sha256": sha256(Path(__file__)),
                  "notes": ["no training or L3", "no configuration promotion", "no threshold selected using test",
                            "empirical validation FPR is not a test/deployment guarantee",
                            "partial L1-L2 negatives would be pending L3, not final benign in full mode11"]}
    write_json(output / "config.json", run_config)
    from models.detectors.layer2_detector import Layer2Detector
    detector = Layer2Detector(**dict(cfg, model_path=source["snapshot"]),
                              device="cpu" if args.device == "cpu" else "cuda:0", local_files_only=True)
    scored = detector.predict([row["sequence"] for row in validation], batch_size=16)
    val_probs = scored["probabilities"]
    val_labels = np.array([row["label"] for row in validation])
    with (output / "validation_predictions.jsonl").open("w") as stream:
        for i, row in enumerate(validation):
            stream.write(json.dumps({"id": row["id"], "label": row["label"], "l1_error": float(val_errors[i]),
                "l2_probability": float(val_probs[i]), "channels": scored["channel_probabilities"][i]}) + "\n")
    scopes = [None] + source["thresholds"]
    frozen = []
    for l1 in scopes:
        eligible = np.ones(len(validation), dtype=bool) if l1 is None else val_errors > l1
        for budget in BUDGETS:
            chosen = select_fpr(val_labels[eligible], val_probs[eligible], budget)
            frozen.append({"l1_threshold": l1, "scope": "standalone_validation" if l1 is None else "L1_routed_validation", **chosen})
    write_json(output / "selection_frozen.json", {"selected_before_loading_test_predictions": True,
        "validation_scores_sha256": sha256(output / "validation_predictions.jsonl"), "selections": frozen})
    print("FROZEN validation-only selections", flush=True)
    # Do not load new test outcomes until all operating points are frozen above.
    test_path = ROOT / "data/processed/test_processed.jsonl"
    if sha256(test_path) != source["test_sha256"]:
        raise ValueError("Test data changed")
    test_rows, test_scores = read_jsonl(test_path), read_jsonl(args.source / "predictions.jsonl")
    if len(test_rows) != len(test_scores) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(test_rows, test_scores)):
        raise ValueError("Test ID/label alignment failed")
    test_labels = np.array([row["label"] for row in test_scores])
    test_errors = np.array([row["l1_error"] for row in test_scores])
    test_probs = np.array([row["l2_probability"] for row in test_scores])
    validation_grid = [evaluate_point(val_labels, val_errors, val_probs, l1, l2) for l1 in scopes for l2 in GRID]
    test_grid = [evaluate_point(test_labels, test_errors, test_probs, l1, l2) for l1 in scopes for l2 in GRID]
    comparisons = []
    for item in frozen:
        l1, l2 = item["l1_threshold"], item["threshold"]
        comparison = {"selection": item,
            "validation": evaluate_point(val_labels, val_errors, val_probs, l1, l2),
            "test": evaluate_point(test_labels, test_errors, test_probs, l1, l2), "test_per_scenario": {}}
        for kind in sorted({row.get("attack_type") or "benign" for row in test_rows}):
            mask = np.array([(row.get("attack_type") or "benign") == kind for row in test_rows])
            comparison["test_per_scenario"][kind] = evaluate_point(test_labels[mask], test_errors[mask], test_probs[mask], l1, l2)
        comparisons.append(comparison)
    write_csv(output / "validation_grid.csv", validation_grid)
    write_csv(output / "test_grid.csv", test_grid)
    write_csv(output / "selected_test.csv", [{"validation_FPR_budget": row["selection"]["fpr_budget"], **row["test"]} for row in comparisons])
    write_json(output / "report.json", {"validation_samples": len(validation), "test_samples": len(test_rows),
        "validation_unique_payloads": scored["unique_payloads"], "validation_windows": scored["windows"],
        "validation_grid": validation_grid, "test_grid": test_grid, "validation_selected_points": comparisons,
        "test_data_sha256": sha256(test_path), "cached_test_scores_sha256": sha256(args.source / "predictions.jsonl")})
    print(f"COMPLETED {output}", flush=True)


if __name__ == "__main__":
    main()
