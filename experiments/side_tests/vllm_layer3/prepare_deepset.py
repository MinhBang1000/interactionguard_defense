"""Freeze Deepset mean-window routing with unchanged few-shot L3 inputs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from .run import digest


def main():
    base = Path("results/side_l3_fewshot/20261021_101634").resolve()
    detector = Path("results/side_detector_survey/20261021_094305/deepset").resolve()
    cfg = json.loads((base/"config.json").read_text())
    selected = json.loads((detector/"selection_frozen.json").read_text())["selections"]["mean"]["0.01"]
    inputs = [json.loads(x) for x in (base/"inputs.jsonl").read_text().splitlines()]
    scores = [json.loads(x) for x in (detector/"test_predictions.jsonl").read_text().splitlines()]
    if len(inputs) != len(scores) or any(a["id"] != b["id"] or a["label"] != b["label"] for a,b in zip(inputs,scores)):
        raise ValueError("L2 scores not aligned")
    for row, prediction in zip(inputs,scores):
        row["l2_probability"] = prediction["mean"]
        row.pop("channels", None)  # Previous ProtectAI attribution is no longer applicable.
    required = {hashlib.sha256(row["payload"].encode()).hexdigest() for row in inputs
                if row["l1_error"] > min(cfg["thresholds"]) and row["l2_probability"] <= selected["threshold"]}
    prior = [json.loads(x) for x in (base/"l3_outputs.jsonl").read_text().splitlines()]
    reusable = [x for x in prior if x["payload_sha256"] in required]
    out = Path("results/side_deepset_l3") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    for name in ["system_prompt.txt", "synthetic_examples.json", "requirements.lock.txt"]:
        shutil.copy2(base/name,out/name)
    for name, rows in [("inputs.jsonl",inputs),("seed_cache.jsonl",reusable)]:
        (out/name).write_text("".join(json.dumps(row)+'\n' for row in rows))
    cfg.update(protocol="exploratory_deepset_mean_with_fewshot_L3", l2_results=str(detector),
        l2_model="deepset/deberta-v3-base-injection", l2_policy="transport JSON decode; mean within payload; max across channels",
        l2_threshold=selected["threshold"], l2_predictions_sha256=digest(detector/"test_predictions.jsonl"),
        l2_selection_sha256=digest(detector/"selection_frozen.json"),
        input_sha256=digest(out/"inputs.jsonl"), seed_cache_sha256=digest(out/"seed_cache.jsonl"),
        cache_source_run=str(base), cache_source_outputs_sha256=digest(base/"l3_outputs.jsonl"),
        baseline_run=str(base), changes=["Deepset JSON mean-window calibrated L2 replaces raw-max ProtectAI .5"],
        limitations=["exploratory repeatedly inspected test; Deepset selected after test inspection",
                     "L2 model, representation, aggregation and threshold differ together",
                     "same clipped few-shot L3 payloads and original L1 scores",
                     "exact matching cached L3 payloads reused; not independent repeats",
                     "no main promotion or production latency claim"])
    (out/"config.json").write_text(json.dumps(cfg,indent=2)+'\n')
    print(json.dumps({"run": str(out),"unique_required":len(required),"reused":len(reusable),"new":len(required)-len(reusable)}))


if __name__ == "__main__":
    main()
