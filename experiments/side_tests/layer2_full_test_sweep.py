"""Standalone released-L2 sweep with identical decoded channel-wise inputs.

No L1, L3, training, downloads, or active configuration changes. Test sweeps are
exploratory; deployment operating points are calibrated on validation only.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np

from .detector_survey import REGISTRY, malicious_index, score
from .layer2_threshold_sweep import select_fpr, write_csv

THRESHOLDS = [0., .1, .2, .3, .4, .5, .6, .7, .8, .85, .9, .95,
              .97, .98, .99, .995, .999, .9995, .9999, .99995,
              .99999, .999995, .999999, 1.]
BUDGETS = [.01, .005, .001]
MODELS = [("protectai_v2", "protectai/deberta-v3-base-prompt-injection-v2",
           "90c9989b1a342275dd0d1a95aad283c04e075671")] + REGISTRY


def summarize(labels, probabilities, threshold):
    from training.layer1_common import metrics
    result = metrics(labels, probabilities, threshold)
    return {**result, "threshold": float(threshold),
            "fnr": result["fn"] / (result["tp"] + result["fn"]) if result["tp"] + result["fn"] else None,
            "precision_defined": bool(result["tp"] + result["fp"]),
            "recall_defined": bool(result["tp"] + result["fn"])}


def verify_cached_inputs(archived_rows, current_rows):
    if len(archived_rows) != len(current_rows):
        raise ValueError("Cached input row count changed")
    for old, new in zip(archived_rows, current_rows):
        for field in ("id", "sequence", "label"):
            if old.get(field) != new.get(field):
                raise ValueError(f"Cached detector input changed: {new.get('id')}/{field}")


def run_cached(output):
    import hashlib
    from zipfile import ZipFile
    from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
    from data.prepare_training import load_prepared
    from training.layer1_common import directory_hash

    output.mkdir(parents=True, exist_ok=False)
    survey = ROOT / "results/side_detector_survey/20261021_094305"
    protect = ROOT / "results/side_layer2_input_ablation/20261021_083614"
    protocol = json.loads((survey / "protocol.json").read_text())
    protect_config = json.loads((protect / "config.json").read_text())
    if sha256(Path(__file__).with_name("detector_survey.py")) != protocol["source_sha256"]:
        raise ValueError("Survey scoring source changed")
    if sha256(Path(__file__).with_name("layer2_input_ablation.py")) != protect_config["source_sha256"]:
        raise ValueError("Decoded-window scoring source changed")
    val = load_prepared("encoder", "val")
    test = read_jsonl(ROOT / "data/processed/test_processed.jsonl")
    backup = ROOT / "backups/data_asr_20261021_202832_743928.zip"
    with ZipFile(backup) as archive:
        for key, path, current in [("validation_sha256", "data/training/encoder/val.jsonl", val),
                                   ("test_sha256", "data/processed/test_processed.jsonl", test)]:
            content = archive.read(path)
            if hashlib.sha256(content).hexdigest() != protocol[key] or protocol[key] != protect_config[key]:
                raise ValueError("Cache and archived dataset identities disagree")
            verify_cached_inputs([json.loads(line) for line in content.decode().splitlines()], current)
    # Cached scores must use the same channel mapping and window implementation.
    with ZipFile(protect / "source_snapshot.zip") as archive:
        for name, path in [("channel_payloads.py", ROOT / "models/detectors/channel_payloads.py"),
                           ("layer2_detector.py", ROOT / "models/detectors/layer2_detector.py")]:
            if archive.read(name) != path.read_bytes():
                raise ValueError(f"Cached inference source changed: {name}")
    manifest = json.loads((survey / "manifest.json").read_text())
    entries = {m["key"]: m for m in manifest["models"]}
    fixed, calibrated, scenarios, completed, cache_records = [], [], [], [], []
    labels_val = np.array([r["label"] for r in val])
    labels_test = np.array([r["label"] for r in test])
    for key, repo, revision in MODELS:
        source = protect if key == "protectai_v2" else survey / key
        if key != "protectai_v2":
            entry = entries[key]
            if entry["repo"] != repo or entry["revision"] != revision:
                raise ValueError("Cached checkpoint revision differs")
            if directory_hash(entry["snapshot"]) != entry["checkpoint_sha256"]:
                raise ValueError("Cached model files changed")
        predictions = {}
        for split, current in [("validation", val), ("test", test)]:
            path = source / f"{split}_predictions.jsonl"
            saved = read_jsonl(path)
            if [(r["id"], r["label"]) for r in current] != [(r["id"], r["label"]) for r in saved]:
                raise ValueError("Cached score IDs/labels changed")
            predictions[split] = {mode: np.array([r["scores"]["json_"+mode] if key == "protectai_v2" else r[mode]
                                                for r in saved]) for mode in ("max", "mean")}
            cache_records.append({"model": key, "split": split, "path": str(path), "sha256": sha256(path)})
        selections = {mode: {str(b): select_fpr(labels_val, predictions["validation"][mode], b) for b in BUDGETS}
                      for mode in ("max", "mean")}
        write_json(output / f"{key}_selection.json", {"validation_only": True, "selections": selections})
        for mode in ("max", "mean"):
            probabilities = predictions["test"][mode]
            for threshold in THRESHOLDS:
                fixed.append({"model": key, "aggregation": mode, **summarize(labels_test, probabilities, threshold)})
            for budget in BUDGETS:
                selection = selections[mode][str(budget)]
                threshold = selection["threshold"]
                calibrated.append({"model": key, "aggregation": mode, "validation_fpr_budget": budget,
                    "validation_precision": selection["validation"]["precision"],
                    "validation_recall": selection["validation"]["recall"],
                    "validation_fpr": selection["validation"]["fpr"],
                    **summarize(labels_test, probabilities, threshold)})
                for kind in ("benign", "prompt", "rag", "tool", "correlated"):
                    mask = np.array([(r.get("attack_type") or "benign") == kind for r in test])
                    scenarios.append({"model": key, "aggregation": mode, "validation_fpr_budget": budget,
                        "scenario": kind, **summarize(labels_test[mask], probabilities[mask], threshold)})
        completed.append(key)
    write_csv(output / "fixed_thresholds.csv", fixed)
    write_csv(output / "validation_calibrated.csv", calibrated)
    write_csv(output / "per_scenario.csv", scenarios)
    write_json(output / "summary.json", {"completed_models": completed, "complete": True,
        "fixed": fixed, "calibrated": calibrated, "per_scenario": scenarios})
    write_json(output / "protocol.json", {"mode": "CPU sweep of verified cached probabilities; no new inference",
        "models": MODELS, "thresholds": THRESHOLDS, "validation_fpr_budgets": BUDGETS,
        "validation_rows": len(val), "test_rows": len(test),
        "validation_sha256": sha256(ROOT / "data/training/encoder/val.jsonl"),
        "test_sha256": sha256(ROOT / "data/processed/test_processed.jsonl"),
        "archived_input_proof": str(backup), "cached_input_ids_sequences_labels_equal": True,
        "caches": cache_records, "source_sha256": sha256(Path(__file__)),
        "representation": "same JSON-decoded channel-wise512token/64overlap; AGENT excluded",
        "limitations": ["exploratory repeatedly inspected test; not a fresh inference repeat",
                        "input metadata changed but verified detector text and labels are identical",
                        "no L1/L3 calls or active config changes",
                        "validation FPR budget is not a test FPR guarantee",
                        "common recipe need not be each model's optimal native configuration"]})
    print("CACHE_SWEEP_COMPLETE " + str(output), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-only", action="store_true", help="Verify archived inputs, then sweep existing scores on CPU")
    args = parser.parse_args(argv)
    if args.cache_only:
        run_cached(args.output_dir.resolve())
        return
    os.environ.update(CUDA_VISIBLE_DEVICES="0", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    from types import SimpleNamespace
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
    from data.prepare_training import load_prepared
    from training.layer1_common import directory_hash
    from training.layer1_pipeline import gpu_preflight
    from .layer2_input_ablation import WindowScorer

    gpu = gpu_preflight("0")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    val = load_prepared("encoder", "val")
    labels_val = np.array([r["label"] for r in val])
    test_path = ROOT / "data/processed/test_processed.jsonl"
    test = read_jsonl(test_path)
    labels_test = np.array([r["label"] for r in test])
    write_json(output / "protocol.json", {
        "models": MODELS, "thresholds": THRESHOLDS, "validation_fpr_budgets": BUDGETS,
        "representation": "channel_payloads + transport JSON decoding; AGENT excluded",
        "window_tokens": 512, "overlap_tokens": 64,
        "policies": ["max_within_payload_then_max_across_payloads", "mean_within_payload_then_max_across_payloads"],
        "comparison": "strict_probability_greater_than_threshold",
        "validation_rows": len(val), "test_rows": len(test),
        "validation_sha256": sha256(ROOT / "data/training/encoder/val.jsonl"),
        "test_sha256": sha256(test_path), "gpu": gpu,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "source_sha256": {str(p): sha256(ROOT / p) for p in [
            "experiments/side_tests/layer2_full_test_sweep.py",
            "experiments/side_tests/detector_survey.py",
            "experiments/side_tests/layer2_input_ablation.py",
            "experiments/side_tests/layer2_threshold_sweep.py",
            "models/detectors/channel_payloads.py"]},
        "limitations": ["exploratory repeatedly inspected test", "no model training or L1/L3 calls",
                        "common input recipe is not necessarily each model's optimal native configuration",
                        "same numeric threshold does not imply calibrated score equivalence",
                        "empirical validation FPR is not a test FPR guarantee"]})
    fixed, calibrated, scenarios, completed = [], [], [], []
    for key, repo, revision in MODELS:
        folder = output / key
        folder.mkdir()
        print("START " + key, flush=True)
        snapshot = snapshot_download(repo, revision=revision, local_files_only=True)
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        if config.num_labels != 2:
            raise ValueError(f"Expected a binary checkpoint: {key}")
        index, mapping = malicious_index(config.id2label, key)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError(f"Missing safe padding token: {key}")
            tokenizer.pad_token = tokenizer.eos_token
        if config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        kwargs = {}
        if config.model_type in {"modernbert", "qwen3"}:
            kwargs["attn_implementation"] = "sdpa"
        if config.model_type == "modernbert":
            config.reference_compile = False
        model = AutoModelForSequenceClassification.from_pretrained(
            snapshot, config=config, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, dtype=torch.float32, **kwargs).to("cuda:0").eval()
        model.requires_grad_(False)
        detector = SimpleNamespace(tokenizer=tokenizer, model=model, max_length=512,
            window_overlap=64, device=torch.device("cuda:0"), malicious_index=index)
        scorer = WindowScorer(detector)
        write_json(folder / "model_config.json", {"repo": repo, "revision": revision,
            "checkpoint_sha256": directory_hash(snapshot), "malicious_index": index,
            "mapping_source": mapping, "hf": config.to_dict()})
        start = time.perf_counter()
        validation = score(val, scorer, folder / "validation_predictions.jsonl")
        selections = {mode: {str(b): select_fpr(labels_val, validation[mode], b) for b in BUDGETS}
                      for mode in ("max", "mean")}
        write_json(folder / "selection_frozen.json", {"validation_only": True, "selections": selections})
        tested = score(test, scorer, folder / "test_predictions.jsonl")
        for mode in ("max", "mean"):
            for threshold in THRESHOLDS:
                fixed.append({"model": key, "aggregation": mode, **summarize(labels_test, tested[mode], threshold)})
            for budget in BUDGETS:
                selection = selections[mode][str(budget)]
                threshold = selection["threshold"]
                calibrated.append({"model": key, "aggregation": mode, "validation_fpr_budget": budget,
                    "validation_precision": selection["validation"]["precision"],
                    "validation_recall": selection["validation"]["recall"],
                    "validation_fpr": selection["validation"]["fpr"],
                    **summarize(labels_test, tested[mode], threshold)})
                for kind in ("benign", "prompt", "rag", "tool", "correlated"):
                    mask = np.array([(r.get("attack_type") or "benign") == kind for r in test])
                    scenarios.append({"model": key, "aggregation": mode, "validation_fpr_budget": budget,
                        "scenario": kind, **summarize(labels_test[mask], tested[mode][mask], threshold)})
        write_json(folder / "runtime.json", {"seconds_including_validation_and_test_scoring": time.perf_counter()-start,
            "unique_decoded_payloads": len(scorer.cache), "not_pipeline_latency": True})
        completed.append(key)
        write_csv(output / "fixed_thresholds.csv", fixed)
        write_csv(output / "validation_calibrated.csv", calibrated)
        write_csv(output / "per_scenario.csv", scenarios)
        write_json(output / "summary.json", {"completed_models": completed, "fixed": fixed,
            "calibrated": calibrated, "per_scenario": scenarios, "complete": len(completed) == len(MODELS)})
        print("COMPLETED " + key, flush=True)
        del scorer, detector, model, tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    print("ALL_MODELS_COMPLETE " + str(output), flush=True)


if __name__ == "__main__":
    main()
