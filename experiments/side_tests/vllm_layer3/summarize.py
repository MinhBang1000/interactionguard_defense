"""Verify saved cascade predictions and export a compact comparison table."""
import argparse
import csv
import json
from pathlib import Path

from .run import binary_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir
    config = json.loads((run / "config.json").read_text())
    summaries = json.loads((run / "summary.json").read_text())
    if len(summaries) != len(config["thresholds"]):
        raise ValueError("Not all thresholds have completed")
    inputs = [json.loads(line) for line in (run / "inputs.jsonl").read_text().splitlines()]
    exports = []
    for summary in summaries:
        threshold = summary["l1_threshold"]
        predictions = [json.loads(line) for line in (run / f"threshold_{threshold:.8f}_predictions.jsonl").read_text().splitlines()]
        if len(predictions) != len(inputs) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(inputs, predictions)):
            raise ValueError("Final predictions are not aligned")
        actual = binary_metrics([row["label"] for row in predictions], [row["prediction"] for row in predictions])
        if actual != summary["full_cascade"]:
            raise ValueError("Reported metrics do not match final predictions")
        l1_l2 = binary_metrics([row["label"] for row in inputs],
            [int(row["l1_error"] > threshold and row["l2_probability"] > config["l2_threshold"]) for row in inputs])
        exports.append({"l1_threshold": threshold, "l1_l2_f1": l1_l2["f1"],
                        **actual, "l3_processed": summary["l3_pending"],
                        "l3_tp": summary["l3_subset"]["tp"], "l3_fp": summary["l3_subset"]["fp"],
                        "l1_bypassed_positive": sum(row["label"] == 1 and row["l1_error"] <= threshold for row in inputs)})
    with (run / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)
    (run / "verification.json").write_text(json.dumps({"all_thresholds_complete": True,
        "aligned_predictions_per_threshold": len(inputs), "metrics_recomputed_match": True}, indent=2))
    print(json.dumps(exports, indent=2))


if __name__ == "__main__":
    main()
