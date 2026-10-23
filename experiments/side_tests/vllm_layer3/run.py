"""Isolated vLLM inference on frozen L1/L2 pending prefixes (GPU0 default)."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def binary_metrics(labels, predictions):
    import numpy as np
    y, p = np.asarray(labels), np.asarray(predictions)
    tp, fp = int(((y == 1) & (p == 1)).sum()), int(((y == 0) & (p == 1)).sum())
    fn, tn = int(((y == 1) & (p == 0)).sum()), int(((y == 0) & (p == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp/(tp+fp) if tp+fp else 0., "recall": tp/(tp+fn) if tp+fn else 0.,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
            "fpr": fp/(fp+tn) if fp+tn else 0.}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--device", choices=["0", "1"], default="0",
                        help="Physical GPU; use a nondefault device only with explicit authorization")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    run = args.run_dir
    config = json.loads((run / "config.json").read_text())
    if int(args.device) != int(config.get("physical_gpu", 0)):
        raise ValueError("Requested GPU differs from the frozen run configuration")
    if digest(run / "inputs.jsonl") != config["input_sha256"] or digest(run / "system_prompt.txt") != config["system_prompt_sha256"]:
        raise ValueError("Frozen inputs or prompt changed")
    if config.get("seed_cache_sha256"):
        source = Path(config["cache_source_run"])
        source_config = json.loads((source / "config.json").read_text())
        for key in ["model", "model_revision", "dtype", "temperature", "seed", "max_tokens", "max_model_len", "schema_variant", "l3_threshold", "system_prompt_sha256"]:
            if config[key] != source_config[key]:
                raise ValueError(f"Cache inference setting differs: {key}")
        if digest(source / "l3_outputs.jsonl") != config["cache_source_outputs_sha256"] or digest(source / "system_prompt.txt") != config["system_prompt_sha256"]:
            raise ValueError("Cache source changed")
        originals = {x["payload_sha256"]: x for x in (json.loads(line) for line in (source / "l3_outputs.jsonl").read_text().splitlines())}
        for record in (json.loads(line) for line in (run / "seed_cache.jsonl").read_text().splitlines()):
            if record != originals.get(record["payload_sha256"]):
                raise ValueError("Seed verdict does not match original cache")
    gpu = subprocess.check_output(["nvidia-smi", "-i", args.device, "--query-gpu=name,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    if "4090" not in gpu or int(gpu.split(",")[-1]) > 1024:
        raise RuntimeError(f"Physical GPU {args.device} must be an idle RTX 4090: {gpu}")
    import numpy as np
    import torch
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    rows = [json.loads(line) for line in (run / "inputs.jsonl").read_text().splitlines()]
    labels = [row["label"] for row in rows]
    system = (run / "system_prompt.txt").read_text()
    bool_keys = ["prompt_attack", "retrieve_attack", "tool_attack", "agent_adoption", "is_malicious"]
    properties = {key: {"type": "boolean"} for key in bool_keys}
    properties.update({key: {"type": "string"} for key in ["user_intent", "evidence_span", "reason", "attack_type", "suspicious_channel"]})
    properties["authorized_actions"] = {"type": "array", "items": {"type": "string"}, "maxItems": 5}
    properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    if config.get("schema_variant") == "authorization_audit":
        for key in ["attack_type", "suspicious_channel", "authorized_actions"]:
            del properties[key]
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    (run / "schema.json").write_text(json.dumps(schema, indent=2))
    (run / "runtime.json").write_text(json.dumps({"vllm": vllm.__version__, "torch": torch.__version__,
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"], "gpu": gpu,
        "source_sha256": digest(__file__), "started": datetime.now(timezone.utc).isoformat()}, indent=2))
    llm = LLM(model=config["model"], dtype=config["dtype"], tensor_parallel_size=1,
        max_model_len=config["max_model_len"], gpu_memory_utilization=config["gpu_memory_utilization"],
        max_num_seqs=config["max_num_seqs"], max_num_batched_tokens=config["max_num_batched_tokens"],
        enable_prefix_caching=True, enable_chunked_prefill=True, seed=config["seed"])
    tokenizer = llm.get_tokenizer()
    params = SamplingParams(temperature=0, max_tokens=config["max_tokens"], seed=config["seed"],
                            guided_decoding=GuidedDecodingParams(json=schema))
    cache_path = run / "l3_outputs.jsonl"
    if cache_path.exists():
        raise ValueError("Output already exists; prepare a new run rather than overwrite")
    cache, summaries = {}, []
    if config.get("seed_cache_sha256"):
        seed_path = run / "seed_cache.jsonl"
        if digest(seed_path) != config["seed_cache_sha256"]:
            raise ValueError("Frozen seed cache changed")
        seeded = {record["payload_sha256"]: record for record in
                  (json.loads(line) for line in seed_path.read_text().splitlines())}
        for row in rows:
            key = hashlib.sha256(row["payload"].encode()).hexdigest()
            if key in seeded:
                record = seeded[key]
                verdict = record["verdict"]
                if set(verdict) != set(properties) or record["finish_reason"] != "stop":
                    raise ValueError("Invalid cached verdict")
                prediction = int(any(verdict[k] for k in bool_keys) and float(verdict["confidence"]) >= config["l3_threshold"])
                if prediction != record["prediction"]:
                    raise ValueError("Cached decision rule differs")
                cache[row["payload"]] = record
        with cache_path.open("w") as stream:
            for record in cache.values():
                stream.write(json.dumps(record) + "\n")
        print(f"REUSED_VERIFIED_PAYLOADS {len(cache)}", flush=True)
    for threshold in config["thresholds"]:
        selected = [i for i, row in enumerate(rows) if row["l1_error"] > threshold and row["l2_probability"] <= config["l2_threshold"]]
        needed = list(dict.fromkeys(rows[i]["payload"] for i in selected if rows[i]["payload"] not in cache))
        print(f"START threshold={threshold:.8f} pending={len(selected)} new_payloads={len(needed)}", flush=True)
        started = time.perf_counter()
        for offset in range(0, len(needed), 64):
            batch = needed[offset:offset+64]
            texts = [tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": payload}], tokenize=False, add_generation_prompt=True) for payload in batch]
            lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
            if any(length + config["max_tokens"] > config["max_model_len"] for length in lengths):
                raise ValueError("A prepared payload exceeds the fixed context budget; no silent truncation")
            outputs = llm.generate(texts, params, use_tqdm=True)
            with cache_path.open("a") as stream:
                for payload, output in zip(batch, outputs):
                    completion = output.outputs[0]
                    if completion.finish_reason == "length":
                        raise ValueError("Truncated JSON completion; do not silently score it as benign")
                    verdict = json.loads(completion.text)
                    if set(verdict) != set(properties) or any(type(verdict[key]) is not bool for key in bool_keys):
                        raise ValueError("Invalid verdict schema")
                    malicious = any(verdict[key] for key in bool_keys)
                    prediction = int(malicious and float(verdict["confidence"]) >= config["l3_threshold"])
                    record = {"payload_sha256": hashlib.sha256(payload.encode()).hexdigest(), "prediction": prediction,
                              "verdict": verdict, "finish_reason": completion.finish_reason,
                              "prompt_tokens": len(output.prompt_token_ids), "completion_tokens": len(completion.token_ids)}
                    cache[payload] = record
                    stream.write(json.dumps(record) + "\n")
        final = [int(row["l1_error"] > threshold and row["l2_probability"] > config["l2_threshold"]) for row in rows]
        for index in selected:
            final[index] = cache[rows[index]["payload"]]["prediction"]
        pending_labels = [labels[i] for i in selected]
        pending_predictions = [final[i] for i in selected]
        summary = {"l1_threshold": threshold, "l2_threshold": config["l2_threshold"], "l3_threshold": config["l3_threshold"],
                   "full_cascade": binary_metrics(labels, final), "l3_subset": binary_metrics(pending_labels, pending_predictions),
                   "l3_pending": len(selected), "new_unique_L3_calls": len(needed),
                   "incremental_wall_seconds_not_pipeline_latency": time.perf_counter()-started,
                   "per_scenario": {kind: binary_metrics([row["label"] for row in rows if row["attack_type"] == kind],
                        [p for row, p in zip(rows, final) if row["attack_type"] == kind]) for kind in sorted({row["attack_type"] for row in rows})}}
        tag = f"threshold_{threshold:.8f}"
        (run / f"{tag}.json").write_text(json.dumps(summary, indent=2))
        with (run / f"{tag}_predictions.jsonl").open("w") as stream:
            for row, prediction in zip(rows, final):
                stream.write(json.dumps({"id": row["id"], "label": row["label"], "prediction": prediction}) + "\n")
        summaries.append(summary)
        (run / "summary.json").write_text(json.dumps(summaries, indent=2))
        print("COMPLETED " + json.dumps(summary), flush=True)
    print("ALL_THRESHOLDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
