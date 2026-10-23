"""Compare released classifiers under the same decoded channel-wise L2 policy."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .layer2_input_ablation import WindowScorer, aggregate_windows, unwrap_json
from .layer2_threshold_sweep import evaluate_point, select_fpr, write_csv

REGISTRY = [
    ("prompt_guard_2", "meta-llama/Llama-Prompt-Guard-2-86M", "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"),
    ("sentinel_v2", "rogue-security/prompt-injection-jailbreak-sentinel-v2", "05ff9bfd1f28c5228e53121d5ee4427bb5205212"),
    ("sentinel_v1", "qualifire/prompt-injection-sentinel", "e24131782819a4787c129ed2b9a4a67ae57a584e"),
    ("proventra", "proventra/mdeberta-v3-base-prompt-injection", "b8a89d3096cf11a71d57b283c854f5ae2ed3df83"),
    ("deepset", "deepset/deberta-v3-base-injection", "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1"),
]
LABEL_REFERENCE = "https://github.com/meta-llama/llama-cookbook/blob/main/getting-started/responsible_ai/prompt_guard/inference.py"


def malicious_index(labels, key):
    matches = [int(index) for index, name in labels.items() if str(name).lower() in {"injection", "malicious", "jailbreak", "unsafe"}]
    if len(matches) == 1:
        return matches[0], "explicit_id2label"
    if key == "prompt_guard_2" and set(str(value).upper() for value in labels.values()) == {"LABEL_0", "LABEL_1"}:
        return 1, LABEL_REFERENCE
    raise ValueError(f"No verified malicious label mapping for {key}: {labels}")


def prepare(output):
    from dotenv import load_dotenv
    from huggingface_hub import snapshot_download
    from data.data_preprocess import ROOT, sha256, write_json
    from training.layer1_common import directory_hash
    load_dotenv(ROOT / ".env")
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"protocol": "released_checkpoints_same_decoded_channelwise_policy", "models": [],
                "source_sha256": sha256(Path(__file__)), "prepared_at": datetime.now(timezone.utc).isoformat()}
    for key, repo, revision in REGISTRY:
        record = {"key": key, "repo": repo, "revision": revision, "card_url": f"https://huggingface.co/{repo}"}
        try:
            snapshot = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "README.md"])
            if not list(Path(snapshot).glob("*.safetensors")):
                raise ValueError("No safetensors weights; will not load pickle or remote code")
            record.update(status="ready", snapshot=snapshot, checkpoint_sha256=directory_hash(snapshot))
        except Exception as exc:
            record.update(status="unavailable", error_type=type(exc).__name__,
                          http_status=getattr(getattr(exc, "response", None), "status_code", None))
        manifest["models"].append(record)
        write_json(output / "manifest.json", manifest)
        print(json.dumps({"model": key, "status": record["status"]}), flush=True)
    print(f"PREPARED {output / 'manifest.json'}", flush=True)


def score(rows, scorer, destination):
    from models.detectors.channel_payloads import channel_payloads
    parts = [channel_payloads(row["sequence"]) for row in rows]
    decoded = {text: unwrap_json(text) for blocks in parts for _, text in blocks}
    scorer.score(list(decoded.values()))
    outputs = {}
    for mode in ["max", "mean"]:
        outputs[mode] = np.array([max((aggregate_windows(scorer.cache[decoded[text]], mode) for _, text in blocks), default=0.) for blocks in parts])
    with destination.open("w") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({"id": row["id"], "label": row["label"], "max": float(outputs["max"][index]), "mean": float(outputs["mean"][index])}) + "\n")
    return outputs


def run(manifest_path):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    from data.data_preprocess import ROOT, read_jsonl, sha256, write_json
    from data.prepare_training import load_prepared
    from training.layer1_common import directory_hash
    from training.layer1_pipeline import gpu_preflight
    gpu_preflight("0")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    manifest = json.loads(manifest_path.read_text())
    output = manifest_path.parent
    if (output / "summary.json").exists():
        raise ValueError("Existing summary; prepare a new run rather than overwrite")
    original = ROOT / "results/layer2_channelwise/20261021_071550_378314"
    source = json.loads((original / "config.json").read_text())
    val = load_prepared("encoder", "val")
    if sha256(ROOT / "data/training/encoder/val.jsonl") != source["validation_sha256"]:
        raise ValueError("Validation changed")
    l1_run = Path(source["l1_run_dir"])
    l1_selected = json.loads((l1_run / "autoencoder/selection_frozen.json").read_text())
    val_error_path = Path(l1_selected["best"]["checkpoint"]).parent / "validation_errors.npy"
    if sha256(val_error_path) != source["validation_errors_sha256"]:
        raise ValueError("Validation errors changed")
    val_errors = np.load(val_error_path)
    val_labels = np.array([row["label"] for row in val])
    write_json(output / "protocol.json", {"window_tokens": 512, "overlap_tokens": 64,
        "payload_policy": "same channel_payloads + unwrap_json; AGENT excluded; all prompt/retrieval/tool sources kept",
        "aggregation": "max or mean within payload, max/OR across payloads/channels", "dtype": "float32",
        "validation_FPR_budgets": [.01, .005, .001], "primary_comparison_budget": .01,
        "selection": "validation only; main summary uses same mean policy for every classifier",
        "no_training_or_L3_or_main_changes": True, "test_scope": "exploratory repeatedly inspected test",
        "validation_sha256": source["validation_sha256"], "test_sha256": source["test_sha256"],
        "torch": str(torch.__version__), "transformers": str(transformers.__version__),
        "source_sha256": sha256(Path(__file__)), "physical_GPU": 0})
    summaries, per_model = [], {}
    for entry in manifest["models"]:
        key = entry["key"]
        if entry["status"] != "ready":
            continue
        folder = output / key
        folder.mkdir(exist_ok=False)
        print("START " + key, flush=True)
        try:
            if directory_hash(entry["snapshot"]) != entry["checkpoint_sha256"]:
                raise ValueError("Classifier files changed")
            cfg = AutoConfig.from_pretrained(entry["snapshot"], local_files_only=True, trust_remote_code=False)
            if cfg.num_labels != 2:
                raise ValueError("Survey requires binary released classifier")
            index, mapping_source = malicious_index(cfg.id2label, key)
            kwargs = {}
            if cfg.model_type in {"modernbert", "qwen3"}:
                kwargs["attn_implementation"] = "sdpa"
            if cfg.model_type == "modernbert":
                cfg.reference_compile = False
            tokenizer = AutoTokenizer.from_pretrained(entry["snapshot"], local_files_only=True, trust_remote_code=False)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError("No safe padding token")
                tokenizer.pad_token = tokenizer.eos_token
            if cfg.pad_token_id is None:
                cfg.pad_token_id = tokenizer.pad_token_id
            model = AutoModelForSequenceClassification.from_pretrained(entry["snapshot"], config=cfg,
                local_files_only=True, trust_remote_code=False, use_safetensors=True, dtype=torch.float32, **kwargs).to("cuda:0").eval()
            model.requires_grad_(False)
            detector = SimpleNamespace(tokenizer=tokenizer, model=model, max_length=512, window_overlap=64,
                                       device=torch.device("cuda:0"), malicious_index=index)
            scorer = WindowScorer(detector)
            write_json(folder / "model_config.json", {"hf": cfg.to_dict(), "malicious_index": index,
                "mapping_source": mapping_source, "model_parameters": sum(p.numel() for p in model.parameters())})
            validation = score(val, scorer, folder / "validation_predictions.jsonl")
            selections = {mode: {str(budget): select_fpr(val_labels, validation[mode], budget) for budget in [.01, .005, .001]} for mode in ["max", "mean"]}
            write_json(folder / "selection_frozen.json", {"validation_only": True, "selections": selections,
                "validation_predictions_sha256": sha256(folder / "validation_predictions.jsonl")})
            # Every model freezes both policies before reading test labels/scores.
            processed = ROOT / "data/processed/test_processed.jsonl"
            if sha256(processed) != source["test_sha256"]:
                raise ValueError("Test changed")
            test, l1 = read_jsonl(processed), read_jsonl(original / "predictions.jsonl")
            if len(test) != len(l1) or any(a["id"] != b["id"] or a["label"] != b["label"] for a, b in zip(test, l1)):
                raise ValueError("Test and L1 IDs not aligned")
            labels = np.array([row["label"] for row in test])
            errors = np.array([row["l1_error"] for row in l1])
            tested = score(test, scorer, folder / "test_predictions.jsonl")
            report = {"model": key, "fixed": [], "calibrated": [], "test_per_scenario": {}}
            for mode in ["max", "mean"]:
                for gate in [None] + source["thresholds"]:
                    report["fixed"].append({"aggregation": mode, **evaluate_point(labels, errors, tested[mode], gate, .5)})
                    for budget in [.01, .005, .001]:
                        threshold = selections[mode][str(budget)]["threshold"]
                        point = {"aggregation": mode, "validation_FPR_budget": budget,
                                 **evaluate_point(labels, errors, tested[mode], gate, threshold)}
                        report["calibrated"].append(point)
                        if gate is None:
                            summaries.append({"model": key, "validation_recall": selections[mode][str(budget)]["validation"]["recall"],
                                "validation_fpr": selections[mode][str(budget)]["validation"]["fpr"], **point})
                report["test_per_scenario"][mode] = {}
                for kind in sorted({row.get("attack_type") or "benign" for row in test}):
                    mask = np.array([(row.get("attack_type") or "benign") == kind for row in test])
                    report["test_per_scenario"][mode][kind] = evaluate_point(labels[mask], errors[mask], tested[mode][mask], None, selections[mode]["0.01"]["threshold"])
            write_json(folder / "report.json", report)
            per_model[key] = report
            write_json(output / "summary.json", {"completed_models": list(per_model), "standalone_calibrated": summaries})
            write_csv(output / "summary.csv", summaries)
            print("COMPLETED " + key + " " + json.dumps([r for r in summaries if r["model"] == key and r["aggregation"] == "mean" and r["validation_FPR_budget"] == .01]), flush=True)
            del scorer, detector, model, tokenizer
            gc.collect()
            torch.cuda.empty_cache()
        except Exception as exc:
            write_json(folder / "error.json", {"error_type": type(exc).__name__, "message": str(exc)[:1600]})
            print("FAILED " + key + " " + type(exc).__name__, flush=True)
            # End this process: do not load another model into a potentially failed CUDA context.
            raise
    print("SURVEY_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.prepare:
        output = args.output_dir or Path("results/side_detector_survey") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        prepare(output)
    elif args.manifest:
        run(args.manifest)
    else:
        parser.error("Use --prepare, then --manifest PATH")


if __name__ == "__main__":
    main()
