"""Combine released-model results with the identical-policy ProtectAI baseline."""
import argparse
import json
from pathlib import Path

import numpy as np

from data.data_preprocess import ROOT, read_jsonl, write_json
from .layer2_threshold_sweep import evaluate_point, select_fpr, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir
    manifest = json.loads((run / "manifest.json").read_text())
    completed = json.loads((run / "summary.json").read_text())
    expected = {row["key"] for row in manifest["models"] if row["status"] == "ready"}
    if expected != set(completed["completed_models"]):
        raise ValueError("Not all available candidate models completed")
    base = ROOT / "results/side_layer2_input_ablation/20261021_083614"
    source = json.loads((ROOT / "results/layer2_channelwise/20261021_071550_378314/config.json").read_text())
    val, test = read_jsonl(base / "validation_predictions.jsonl"), read_jsonl(base / "test_predictions.jsonl")
    l1 = read_jsonl(ROOT / "results/layer2_channelwise/20261021_071550_378314/predictions.jsonl")
    assert len(test) == len(l1) and all(a["id"] == b["id"] and a["label"] == b["label"] for a, b in zip(test, l1))
    val_labels, labels = np.array([r["label"] for r in val]), np.array([r["label"] for r in test])
    errors = np.array([r["l1_error"] for r in l1])
    standalone, combinations, fixed, validation = [], [], [], []
    for mode in ["max", "mean"]:
        name = "json_" + mode
        v = np.array([r["scores"][name] for r in val])
        t = np.array([r["scores"][name] for r in test])
        chosen = select_fpr(val_labels, v, .01)
        row = {"model": "protectai_v2", "aggregation": mode, "validation_FPR_budget": .01,
               "validation_recall": chosen["validation"]["recall"], "validation_fpr": chosen["validation"]["fpr"],
               **evaluate_point(labels, errors, t, None, chosen["threshold"])}
        standalone.append(row)
        validation.append({"model": "protectai_v2", "aggregation": mode, **chosen})
        fixed.append({"model": "protectai_v2", "aggregation": mode, **evaluate_point(labels, errors, t, None, .5)})
        for gate in source["thresholds"]:
            combinations.append({"model": "protectai_v2", "aggregation": mode,
                                 **evaluate_point(labels, errors, t, gate, chosen["threshold"])})
    for entry in manifest["models"]:
        if entry["status"] != "ready":
            continue
        key = entry["key"]
        report = json.loads((run / key / "report.json").read_text())
        selections = json.loads((run / key / "selection_frozen.json").read_text())["selections"]
        val_rows = read_jsonl(run / key / "validation_predictions.jsonl")
        test_rows = read_jsonl(run / key / "test_predictions.jsonl")
        assert len(val_rows) == len(val) and all(a["id"] == b["id"] and a["label"] == b["label"] for a, b in zip(val_rows, val))
        assert len(test_rows) == len(test) and all(a["id"] == b["id"] and a["label"] == b["label"] for a, b in zip(test_rows, test))
        for mode in ["max", "mean"]:
            chosen = selections[mode]["0.01"]
            validation.append({"model": key, "aggregation": mode, **chosen})
            fixed.extend({"model": key, **r} for r in report["fixed"] if r["aggregation"] == mode and r["l1_threshold"] is None)
            for r in report["calibrated"]:
                if r["aggregation"] != mode or r["validation_FPR_budget"] != .01:
                    continue
                if r["l1_threshold"] is None:
                    standalone.append({"model": key, "validation_recall": chosen["validation"]["recall"],
                                       "validation_fpr": chosen["validation"]["fpr"], **r})
                else:
                    combinations.append({"model": key, **r})
    preferred = max([row for row in validation if row["aggregation"] == "mean"],
                    key=lambda row: (row["validation"]["tp"], -row["validation"]["fp"]))
    # CSV rows from baseline/candidates have the same meanings but optional budget fields.
    for rows, filename in [(standalone, "comparison.csv"), (combinations, "l1_comparison.csv"), (fixed, "fixed_05.csv")]:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        write_csv(run / filename, [{key: row.get(key) for key in fields} for row in rows])
    result = {"primary_aggregation": "mean", "selection_budget": .01, "selected_by_validation": preferred,
              "validation": validation, "standalone": standalone, "l1_l2": combinations, "fixed_05": fixed,
              "notes": ["all decoded channel-wise policy", "selection on validation only", "test exploratory",
                        "no full cascade, L3 calls, ASR or promotion"]}
    write_json(run / "comparison.json", result)
    print(json.dumps({"selected_by_validation": preferred,
                      "mean_comparison": [row for row in standalone if row["aggregation"] == "mean"]}, indent=2))


if __name__ == "__main__":
    main()
