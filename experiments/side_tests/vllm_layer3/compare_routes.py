"""Compare saved cascades with different L2 routes but identical L1/L3 inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from .run import binary_metrics


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    run = parser.parse_args().run_dir
    config = json.loads((run/"config.json").read_text())
    baseline = Path(config["baseline_run"])
    old_inputs, inputs = read_rows(baseline/"inputs.jsonl"), read_rows(run/"inputs.jsonl")
    if len(old_inputs) != len(inputs) or any(any(a[k] != b[k] for k in ["id","label","l1_error","payload"]) for a,b in zip(old_inputs,inputs)):
        raise ValueError("L1, labels or L3 payloads differ")
    old = json.loads((baseline/"summary.json").read_text())
    summaries = json.loads((run/"summary.json").read_text())
    if len(summaries) != len(config["thresholds"]):
        raise ValueError("Incomplete run")
    cache = {x["payload_sha256"]: x for x in read_rows(run/"l3_outputs.jsonl")}
    output = []
    for summary in summaries:
        gate = summary["l1_threshold"]
        saved = read_rows(run/f"threshold_{gate:.8f}_predictions.jsonl")
        predicted, subset_y, subset_p = [], [], []
        for row in inputs:
            pred = 0
            if row["l1_error"] > gate:
                if row["l2_probability"] > config["l2_threshold"]:
                    pred = 1
                else:
                    record = cache[hashlib.sha256(row["payload"].encode()).hexdigest()]
                    v = record["verdict"]
                    pred = int(any(v[k] for k in ["prompt_attack","retrieve_attack","tool_attack","agent_adoption","is_malicious"]) and v["confidence"] >= config["l3_threshold"])
                    subset_y.append(row["label"])
                    subset_p.append(pred)
            predicted.append(pred)
        if predicted != [x["prediction"] for x in saved] or binary_metrics(subset_y,subset_p) != summary["l3_subset"]:
            raise ValueError("Saved verdicts do not reproduce routing results")
        previous = next(x for x in old if x["l1_threshold"] == gate)
        for label, s in [("protectai_raw_max_0.5", previous),("deepset_decoded_mean_calibrated",summary)]:
            output.append({"l1_threshold":gate,"l2_variant":label,**s["full_cascade"],
                           "l3_pending":s["l3_pending"],"l3_tp":s["l3_subset"]["tp"],"l3_fp":s["l3_subset"]["fp"]})
    with (run/"route_comparison.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    result={"rows":output,"all_routes_recomputed_from_verdicts":True,
            "notes":["combined L2 model, preprocessing, pooling and threshold comparison; not model-only ablation",
                     "same frozen L1, L3 prompt and clipped payloads; inspected test exploratory"]}
    (run/"route_comparison.json").write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
