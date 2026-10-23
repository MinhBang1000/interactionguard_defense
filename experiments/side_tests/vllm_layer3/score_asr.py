"""Score stored side-test predictions using the shared offline episode ASR contract."""
import argparse
import csv
import json
from pathlib import Path

from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
from eval.asr import evaluate_files, summarize


def compact(report):
    keys = ("total", "success", "unknown", "asr", "asr_lower_bound", "asr_upper_bound",
            "baseline_successful", "timely_prevented", "prevention_rate_on_baseline_successes",
            "naturally_failed", "pre_injection_blocks", "payload_observed")
    known_types = [row for row in report["episodes"] if row["attack_type"] in {"rag", "tool", "correlated"}]
    pooled = summarize(known_types)["global"]
    return {"per_attack_type": {kind: {key: stats[key] for key in keys}
                               for kind, stats in report["per_attack_type"].items()},
            "three_type_pooled": {key: pooled[key] for key in keys},
            "all_attack_types": {key: report["global"][key] for key in keys},
            "benign": report["benign"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--judges", type=Path, default=ROOT / "eval/prompt_success_judgements.jsonl")
    args = parser.parse_args()
    run = args.run_dir
    config = json.loads((run / "config.json").read_text())
    processed = ROOT / "data/processed/test_processed.jsonl"
    traces = ROOT / "data/splits/test.jsonl"
    if sha256(processed) != config["test_sha256"]:
        raise ValueError("Processed test differs from the side test")
    rows = read_jsonl(processed)
    output = run / "asr"
    output.mkdir(exist_ok=False)
    baseline = evaluate_files(traces, judges=args.judges, output=output / "no_defense.json")
    summaries = {"protocol": "offline_replay_not_live_execution", "timing_rule": baseline["timing_policy"],
                 "ASR_definition": "surviving successful attack episodes / all attack attempts in that category",
                 "unknown_policy": "No missing prompt judge is treated as failure or zero ASR",
                 "source_sha256": sha256(Path(__file__)), "shared_ASR_source_sha256": sha256(ROOT / "eval/asr.py"),
                 "baseline": compact(baseline), "thresholds": []}
    csv_rows = []
    for threshold in config["thresholds"]:
        tag = f"threshold_{threshold:.8f}"
        prediction_path = run / f"{tag}_predictions.jsonl"
        predictions = read_jsonl(prediction_path)
        if len(predictions) != len(rows) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(rows, predictions)):
            raise ValueError("Predictions and processed prefixes are not aligned")
        wrapped = output / f"{tag}_verified_predictions.json"
        write_json(wrapped, {"processed_sha256": sha256(processed),
            "source_predictions_sha256": sha256(prediction_path),
            "predictions": [{"id": row["id"], "pred": row["prediction"]} for row in predictions]})
        report = evaluate_files(traces, predictions=wrapped, processed=processed, judges=args.judges,
                                output=output / f"{tag}.json")
        summary = {"l1_threshold": threshold, **compact(report)}
        summaries["thresholds"].append(summary)
        for kind, stats in {**summary["per_attack_type"], "three_type_pooled": summary["three_type_pooled"],
                            "all_attack_types": summary["all_attack_types"]}.items():
            csv_rows.append({"l1_threshold": threshold, "category": kind, **stats})
    write_json(output / "summary.json", summaries)
    with (output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
