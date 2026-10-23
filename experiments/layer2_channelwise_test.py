"""Offline diagnostic L1/L2 routing sweep. No training and no L3 calls."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

DEFAULT_RUN = "models/layer1/training_runs/long_context_20261021"


def routing_summary(labels, errors, probabilities, l1_threshold, l2_threshold):
    import numpy as np
    from training.layer1_common import metrics

    labels, errors, probabilities = map(np.asarray, (labels, errors, probabilities))
    escalate = errors > l1_threshold
    blocked = escalate & (probabilities > l2_threshold)
    pending = escalate & ~blocked
    summary = {
        "l1_threshold": l1_threshold, "l2_threshold": l2_threshold,
        "l1": metrics(labels, errors, l1_threshold),
        # Mode 10 only: unresolved L2 negatives treated as forwarded, NOT L3 verdicts.
        "l1_l2_only": metrics(labels, blocked.astype(float), 0.5),
        "l2_processed": int(escalate.sum()), "l2_blocked": int(blocked.sum()),
        "l3_pending": int(pending.sum()),
        "l3_pending_positive": int((pending & (labels == 1)).sum()),
        "l3_pending_negative": int((pending & (labels == 0)).sum()),
        "early_accept": int((~escalate).sum()),
    }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0", choices=["0", "1", "cpu"])
    parser.add_argument("--run-dir", type=Path, default=Path(DEFAULT_RUN))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.device == "cpu" else args.device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
    from training.layer1_common import calibrate, directory_hash, metrics, provenance
    from training.layer1_pipeline import gpu_preflight
    if args.device != "cpu":
        gpu_preflight(args.device)
    import numpy as np
    import yaml
    from huggingface_hub import snapshot_download
    from models.detectors.layer2_detector import Layer2Detector

    output = args.output_dir or ROOT / "results/layer2_channelwise" / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output.mkdir(parents=True, exist_ok=False)
    test_path = ROOT / "data/processed/test_processed.jsonl"
    score_path = args.run_dir / "autoencoder/test_predictions.jsonl"
    test_report = json.loads((args.run_dir / "autoencoder/test_report.json").read_text())
    if sha256(test_path) != test_report["test_sha256"]:
        raise ValueError("Test data differs from the frozen L1 test; do not join stale errors")
    rows, scores = read_jsonl(test_path), read_jsonl(score_path)
    if len(rows) != len(scores) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(rows, scores)):
        raise ValueError("L1 score IDs/labels are not aligned with the test data")
    cfg = yaml.safe_load((ROOT / "config/detection_config.yaml").read_text())["layers"]["layer2"]["params"]
    if cfg["model_path"] != "protectai/deberta-v3-base-prompt-injection-v2" or not cfg.get("channel_wise"):
        raise ValueError("This experiment requires original HF ProtectAI v2 with channel-wise routing")
    snapshot = snapshot_download(cfg["model_path"], revision=cfg["revision"], local_files_only=True)
    selection = json.loads((args.run_dir / "autoencoder/selection_frozen.json").read_text())
    validation_path = ROOT / "data/training/encoder/val.jsonl"
    validation_error_path = Path(selection["best"]["checkpoint"]).parent / "validation_errors.npy"
    from data.prepare_training import load_prepared
    encoder_config = json.loads((args.run_dir / "encoder/training_config.json").read_text())
    if encoder_config["prepared_manifest_sha256"] != sha256(ROOT / "data/training/manifest.json"):
        raise ValueError("Prepared validation manifest differs from the trained encoder")
    validation_rows = load_prepared("encoder", "val")
    validation_errors = np.load(validation_error_path)
    thresholds = [calibrate([row["label"] for row in validation_rows], validation_errors, floor)["threshold"]
                  for floor in [.98, .99, .995, .999]]
    config = {"protocol": "exploratory_saved_test_scores_no_training_no_L3",
              "model": cfg, "snapshot": snapshot, "checkpoint_sha256": directory_hash(snapshot),
              "test_sha256": sha256(test_path), "l1_scores_sha256": sha256(score_path),
              "l1_run_dir": str(args.run_dir.resolve()), "device_physical": args.device,
              "thresholds": thresholds, "validation_sha256": sha256(validation_path),
              "validation_errors_sha256": sha256(validation_error_path),
              "source_sha256": {str(path): sha256(ROOT / path) for path in [
                  "experiments/layer2_channelwise_test.py", "models/detectors/layer2_detector.py",
                  "models/detectors/channel_payloads.py"]},
              "thresholds_origin": "prior validation recall floors .98/.99/.995/.999; inspected test diagnostic",
              "window_policy": "512 tokens including specials; overlap 64; max across windows/documents/channels",
              "agent_channel": "excluded", "provenance": provenance()}
    write_json(output / "config.json", config)
    runtime_cfg = dict(cfg, model_path=snapshot)
    detector = Layer2Detector(**runtime_cfg, device="cpu" if args.device == "cpu" else "cuda:0", local_files_only=True)
    started = time.perf_counter()
    result = detector.predict([row["sequence"] for row in rows], batch_size=args.batch_size)
    seconds = time.perf_counter() - started
    probabilities = result["probabilities"]
    labels = np.array([row["label"] for row in rows])
    errors = np.array([score["error"] for score in scores])
    summaries = []
    for threshold in thresholds:
        summary = routing_summary(labels, errors, probabilities, threshold, cfg["threshold"])
        summary["per_scenario"] = {}
        for kind in sorted({row.get("attack_type") or "benign" for row in rows}):
            mask = np.array([(row.get("attack_type") or "benign") == kind for row in rows])
            summary["per_scenario"][kind] = routing_summary(labels[mask], errors[mask], probabilities[mask], threshold, cfg["threshold"])
        summaries.append(summary)
    write_json(output / "report.json", {"samples": len(rows), "l2_standalone": metrics(labels, probabilities, cfg["threshold"]),
        "l2_inference_seconds": seconds, "unique_payloads": result["unique_payloads"], "windows": result["windows"],
        "warning": "Partial L1-L2 metrics are not full-cascade results. Cached/deduplicated time is not pipeline latency or ASR.",
        "thresholds": summaries})
    with (output / "predictions.jsonl").open("w") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], "label": row["label"], "l1_error": float(errors[index]),
                "l2_probability": float(probabilities[index]), "channels": result["channel_probabilities"][index]}) + "\n")
    fields = ["l1_threshold", "precision", "recall", "f1", "fpr", "tp", "fp", "fn", "tn", "l2_processed", "l3_pending", "l3_pending_positive", "l3_pending_negative"]
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: summary["l1_l2_only"].get(field, summary.get(field)) for field in fields})
    print(json.dumps({"output": str(output), "thresholds": summaries}, indent=2))


if __name__ == "__main__":
    main()
