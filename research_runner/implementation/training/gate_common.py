"""Provenance and exact validation-only threshold selection for Gate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import subprocess
from datetime import datetime, timezone

import numpy as np

from research_runner.implementation.data.data_preprocess import ROOT, sha256, write_json


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def provenance():
    import torch
    import transformers
    return {"recorded_at_utc": timestamp(), "python": platform.python_version(),
            "torch": str(torch.__version__), "transformers": str(transformers.__version__),
            "cuda_visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "prepared_manifest_sha256": sha256(ROOT / "data/training/manifest.json")}


def directory_hash(path):
    digest = hashlib.sha256()
    for file in sorted(Path(path).glob("*")):
        if file.is_file() and file.suffix in {".json", ".safetensors"}:
            digest.update(file.name.encode())
            digest.update(sha256(file).encode())
    return digest.hexdigest()


def metrics(labels, errors, threshold):
    labels = np.asarray(labels, dtype=int)
    predictions = np.asarray(errors, dtype=float) > threshold
    tp = int(np.sum((labels == 1) & predictions))
    fp = int(np.sum((labels == 0) & predictions))
    fn = int(np.sum((labels == 1) & ~predictions))
    tn = int(np.sum((labels == 0) & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "fpr": fp/(fp+tn) if fp+tn else 0.0, "samples": len(labels)}


def calibrate(labels, errors, recall_floor=0.98):
    """Strict error > threshold; include tied scores as groups, never split ties."""
    labels, errors = np.asarray(labels, dtype=int), np.asarray(errors, dtype=float)
    if len(labels) != len(errors) or not len(labels) or set(labels) != {0, 1}:
        raise ValueError("Calibration requires aligned, nonempty validation scores of both classes")
    if not np.isfinite(errors).all() or not 0 <= recall_floor <= 1:
        raise ValueError("Invalid errors or recall floor")
    order = np.argsort(errors, kind="stable")
    sorted_errors, sorted_labels = errors[order], labels[order]
    positives, negatives = int(labels.sum()), int((labels == 0).sum())
    thresholds = [float(sorted_errors[0] - max(1e-12, abs(sorted_errors[0])*1e-6))]
    tp_values, fp_values = [positives], [negatives]
    tp, fp = positives, negatives
    for i, error in enumerate(sorted_errors):
        tp -= int(sorted_labels[i] == 1)
        fp -= int(sorted_labels[i] == 0)
        if i == len(labels)-1 or sorted_errors[i+1] != error:
            thresholds.append(float(error)); tp_values.append(tp); fp_values.append(fp)
    candidates = []
    for threshold, tp, fp in zip(thresholds, tp_values, fp_values):
        fn = positives - tp
        f1 = 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
        candidates.append((f1, -fp/negatives, tp/positives, threshold))
    best = max(candidates)
    constrained = max(c for c in candidates if c[2] >= recall_floor)
    return {"objective": "max_validation_f1_subject_to_recall_floor", "recall_floor": recall_floor,
            "threshold": constrained[3], "metrics": metrics(labels, errors, constrained[3]),
            "unconstrained": {"threshold": best[3], "metrics": metrics(labels, errors, best[3])},
            "candidate_thresholds": len(candidates), "comparison": "strict_greater_than"}


def write_run(path, settings):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    write_json(path / "run_config.json", {"settings": settings, "provenance": provenance()})
