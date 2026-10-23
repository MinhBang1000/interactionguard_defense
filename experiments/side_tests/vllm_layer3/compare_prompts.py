"""Verify paired L3 prompt results and export differences without changing routing."""
import argparse
import csv
import json
from pathlib import Path

from .run import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir
    config = json.loads((run / "config.json").read_text())
    baseline = Path(config["baseline_run"])
    if digest(run / "inputs.jsonl") != digest(baseline / "inputs.jsonl"):
        raise ValueError("Paired inputs differ")
    old = json.loads((baseline / "summary.json").read_text())
    new = json.loads((run / "summary.json").read_text())
    if len(new) != len(config["thresholds"]):
        raise ValueError("Run incomplete")
    exports = []
    for current in new:
        previous = next(x for x in old if x["l1_threshold"] == current["l1_threshold"])
        if previous["l3_pending"] != current["l3_pending"]:
            raise ValueError("Routing changed")
        for label, summary in [("baseline", previous), ("candidate", current)]:
            exports.append({"l1_threshold": current["l1_threshold"], "prompt": label,
                            **summary["full_cascade"], "l3_pending": summary["l3_pending"],
                            **{"l3_"+k: v for k, v in summary["l3_subset"].items()}})
    outputs = [json.loads(x) for x in (run / "l3_outputs.jsonl").read_text().splitlines()]
    branches = ["prompt_attack", "retrieve_attack", "tool_attack", "agent_adoption"]
    diagnostics = {
        "unique_audits": len(outputs),
        "final_branch_or_disagreements": sum(x["verdict"]["is_malicious"] != any(x["verdict"][k] for k in branches) for x in outputs),
        "confidence_band_violations": sum(not (.65 <= x["verdict"]["confidence"] <= 1 if x["verdict"]["is_malicious"] else 0 <= x["verdict"]["confidence"] <= .35) for x in outputs),
        "finish_reasons": sorted({x["finish_reason"] for x in outputs}),
        "notes": ["same clipped inputs and old L2 routing", "test exploratory, not validation selection",
                  "same schema; only synthetic demonstrations added" if config.get("protocol") == "exploratory_synthetic_contrastive_fewshot" else "schema and prompt changed together",
                  "OR branch confidence>=0.5 execution rule retained"]}
    with (run / "paired_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(exports[0]))
        writer.writeheader()
        writer.writerows(exports)
    (run / "paired_comparison.json").write_text(json.dumps({"rows": exports, "diagnostics": diagnostics}, indent=2) + "\n")
    print(json.dumps({"rows": exports, "diagnostics": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
