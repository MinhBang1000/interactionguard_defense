"""Freeze new L1 + chosen Deepset routing, retaining the existing L3 policy."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

from data.data_preprocess import ROOT, read_jsonl, write_json, write_jsonl
from models.detectors.layer3_detector_o2 import Layer3DetectorO2
from .run import digest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--l1-run", type=Path, required=True)
    parser.add_argument("--l2-results", type=Path, required=True)
    parser.add_argument("--cache-source", type=Path, default=ROOT / "results/side_deepset_l3/20261021_102414")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["0", "1"], default="0")
    args = parser.parse_args(argv)
    base, output = args.cache_source.resolve(), args.output_dir.resolve()
    config = json.loads((base / "config.json").read_text())
    l2cfg = json.loads((args.l2_results / "config.json").read_text())
    processed = ROOT / "data/processed/test_processed.jsonl"
    l1report = json.loads((args.l1_run / "autoencoder/test_report.json").read_text())
    if digest(processed) != l2cfg["test_sha256"] or digest(processed) != l1report["test_sha256"]:
        raise ValueError("Current test and frozen model score sources differ")
    if digest(ROOT / "models/detectors/layer3_detector_o2.py") != config["source_L3_sha256"]:
        raise ValueError("L3 payload-builder implementation changed")
    if digest(base / "system_prompt.txt") != config["system_prompt_sha256"]:
        raise ValueError("Cache source prompt changed")
    current = read_jsonl(processed)
    old = read_jsonl(base / "inputs.jsonl")
    l1 = read_jsonl(args.l1_run / "autoencoder/test_predictions.jsonl")
    l2path = args.l2_results / "test_predictions.jsonl"
    l2 = read_jsonl(l2path)
    identity = [(r["id"], r["label"]) for r in current]
    for records in (old, l1, l2):
        if identity != [(r["id"], r["label"]) for r in records]:
            raise ValueError("L1/L2/cache prefix IDs or labels differ")
    auditor = Layer3DetectorO2.__new__(Layer3DetectorO2)
    for key in ("max_prompt_chars", "max_block_chars", "max_blocks_per_channel"):
        setattr(auditor, key, config["original_L3_params"][key])
    rows = []
    for current_row, cached_row, l1row, l2row in zip(current, old, l1, l2):
        payload = auditor._build_trace_payload(current_row["sequence"])
        if payload != cached_row["payload"]:
            raise ValueError(f"L3 input changed: {current_row['id']}")
        rows.append({"id": current_row["id"], "label": current_row["label"],
                     "attack_type": current_row.get("attack_type") or "benign",
                     "l1_error": l1row["error"], "l2_probability": l2row["mean"], "payload": payload})
    gates = json.loads((args.l1_run / "thresholds_frozen.json").read_text())["gates"]
    thresholds = [g["threshold"] for g in gates]
    l2threshold = l2cfg["l2_threshold"]
    required = {hashlib.sha256(r["payload"].encode()).hexdigest() for r in rows
                if r["l1_error"] > min(thresholds) and r["l2_probability"] <= l2threshold}
    prior = read_jsonl(base / "l3_outputs.jsonl")
    reusable = [record for record in prior if record["payload_sha256"] in required]
    output.mkdir(parents=True, exist_ok=False)
    for name in ("system_prompt.txt", "synthetic_examples.json", "requirements.lock.txt"):
        shutil.copy2(base / name, output / name)
    write_jsonl(output / "inputs.jsonl", rows)
    write_jsonl(output / "seed_cache.jsonl", reusable)
    config.update(protocol="exploratory_retrained_L1_deepset_same_fewshot_L3",
        l1_run_dir=str(args.l1_run.resolve()), l1_scores_sha256=digest(args.l1_run / "autoencoder/test_predictions.jsonl"),
        l1_selection_sha256=digest(args.l1_run / "autoencoder/selection_frozen.json"),
        l2_results=str(args.l2_results.resolve()), l2_predictions_sha256=digest(l2path),
        l2_selection_sha256=l2cfg["selection_sha256"], l2_threshold=l2threshold,
        test_sha256=digest(processed), input_sha256=digest(output / "inputs.jsonl"),
        thresholds=thresholds, threshold_gates=gates, physical_gpu=int(args.device),
        seed_cache_sha256=digest(output / "seed_cache.jsonl"), cache_source_run=str(base),
        cache_source_outputs_sha256=digest(base / "l3_outputs.jsonl"), baseline_run=str(base),
        changes=["freshly retrained encoder/AE and validation-derived L1 gates",
                 "chosen released Deepset JSON-mean L2 with frozen validation threshold",
                 "explicitly authorized physical GPU1 for missing L3 inference" if args.device == "1" else "physical GPU0"],
        limitations=["one seed; exploratory repeatedly inspected test",
                     "same clipped O2 payload, unchanged few-shot prompt, model and schema",
                     "exact cached L3 verdicts reused; not an independent L3 inference repeat",
                     "few-shot examples authored after earlier test inspection",
                     "no production latency claim or live agent rerun"])
    write_json(output / "config.json", config)
    write_json(output / "preparation_verification.json", {"aligned_prefixes": len(rows),
        "rebuilt_payloads_equal_cached": True, "unique_required": len(required),
        "reused": len(reusable), "new": len(required)-len(reusable)})
    with ZipFile(output / "source_snapshot.zip", "w", ZIP_DEFLATED) as archive:
        for path in [Path(__file__), Path(__file__).with_name("run.py"),
                     Path(__file__).with_name("score_asr.py"), ROOT / "eval/asr.py",
                     ROOT / "models/detectors/layer3_detector_o2.py"]:
            archive.write(path, str(path.resolve().relative_to(ROOT)))
    print(json.dumps({"run": str(output), "required": len(required), "reused": len(reusable),
                      "new": len(required)-len(reusable)}), flush=True)


if __name__ == "__main__":
    main()
