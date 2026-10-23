"""Reproducible long-context encoder -> AE search -> calibration -> sealed test.

Use --promote to activate the validation-selected pair only after successful runs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from data.data_preprocess import ROOT, sha256, write_json


def verify_runtime(run_dir):
    import numpy as np
    import torch
    from data.prepare_training import load_prepared
    from models.detectors.layer1_detector import Layer1Detector

    run_dir = Path(run_dir)
    selection = json.loads((run_dir / "autoencoder/selection_frozen.json").read_text())
    best = selection["best"]
    rows = load_prepared("encoder", "val")
    indices = sorted({0, next(i for i, r in enumerate(rows) if r["label"] == 1),
                      max(range(len(rows)), key=lambda i: len(rows[i]["sequence"]))})
    expected = np.load(Path(best["checkpoint"]).parent / "validation_errors.npy")[indices]
    detector = Layer1Detector(str(run_dir / "encoder"), best["checkpoint"], "cuda:0", best["calibration"]["threshold"])
    embeddings = detector.extract_embedding([rows[i]["sequence"] for i in indices],
                                            batch_size=1, cache_dir=str(run_dir / "runtime_verification_cache"))
    actual = detector.compute_reconstruction_errors(embeddings)
    delta = float(np.max(np.abs(expected-actual)))
    report = {"validation_ids": [rows[i]["id"] for i in indices], "max_absolute_error_delta": delta,
              "tolerance": 1e-5, "passed": bool(np.allclose(expected, actual, rtol=1e-5, atol=1e-5))}
    write_json(run_dir / "runtime_consistency.json", report)
    if not report["passed"]:
        raise ValueError(f"Calibration and runtime disagree: {report}")
    del detector
    torch.cuda.empty_cache()
    return report


def gpu_preflight(device):
    output = subprocess.check_output(["nvidia-smi", "-i", str(device),
        "--query-gpu=index,uuid,name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    fields = [s.strip() for s in output.split(",")]
    if len(fields) != 5 or "4090" not in fields[2]:
        raise RuntimeError(f"Expected the requested RTX 4090, got {output}")
    if int(fields[4]) > 1024:
        raise RuntimeError(f"GPU is not idle; refusing to compete with existing jobs: {output}")
    return dict(zip(("index", "uuid", "name", "memory_total_mib", "memory_used_mib"), fields))


def promote(run_dir, expected_yaml_hash):
    run_dir = Path(run_dir)
    yaml_path = ROOT / "config/detection_config.yaml"
    selection_path = run_dir / "autoencoder/selection_frozen.json"
    selection = json.loads(selection_path.read_text())
    encoder = run_dir / "encoder"
    best = selection["best"]
    encoder_record = json.loads((encoder / "training_config.json").read_text())
    if (encoder / "config.json").exists():
        encoder_record.setdefault("model_config", json.loads((encoder / "config.json").read_text()))
    if (encoder / "tokenizer.json").exists():
        encoder_record.setdefault("tokenizer_sha256", sha256(encoder / "tokenizer.json"))
    checkpoint = Path(best["checkpoint"]).resolve()
    if sha256(checkpoint) != best["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint changed")
    if sha256(yaml_path) != expected_yaml_hash:
        raise ValueError("Detection config changed during training; inspect selection and promote manually")
    profile = {"run_dir": str(run_dir), "selection_source": "validation_only",
               "selection_sha256": sha256(selection_path),
               "encoder_path": str(encoder), "ae_path": str(checkpoint),
               "threshold": best["calibration"]["threshold"],
               "encoder": encoder_record,
               "autoencoder": {**selection["settings"], "learning_rate": best["learning_rate"],
                               "latent_dim": best["latent_dim"], "checkpoint_sha256": best["checkpoint_sha256"]},
               "validation": best["calibration"], "test_used_for_selection": False}
    if (run_dir / "autoencoder/test_report.json").exists():
        report_path = run_dir / "autoencoder/test_report.json"
        profile["test_report"] = {"path": str(report_path), "sha256": sha256(report_path),
                                  "metrics": json.loads(report_path.read_text())["all_operational_labels"]}
    if (run_dir / "runtime_consistency.json").exists():
        profile["runtime_consistency"] = json.loads((run_dir / "runtime_consistency.json").read_text())
    profile["deployment_status"] = "experimental_validation_selected_not_security_certified"
    original = yaml_path.read_text()
    (run_dir / "previous_detection_config.yaml").write_text(original)
    match = re.search(r"(?ms)^  layer1:\n.*?(?=^  layer2:)", original)
    if not match:
        raise ValueError("Cannot locate Layer 1 config block")
    block = match.group()
    for key, value in (("bert_model_path", str(encoder)), ("ae_model_path", str(checkpoint)),
                       ("threshold", best["calibration"]["threshold"])):
        block, count = re.subn(rf"(?m)^(      {key}:).*$", lambda m: m[1]+" "+json.dumps(value), block)
        if count != 1:
            raise ValueError(f"Expected one config field: {key}")
    import yaml
    candidate = original[:match.start()] + block + original[match.end():]
    parsed = yaml.safe_load(candidate)
    if parsed["layers"]["layer1"]["params"]["threshold"] != profile["threshold"]:
        raise ValueError("Deployment config validation failed")
    profile_path = ROOT / "config/layer1_best.json"
    if profile_path.exists():
        (run_dir / "previous_layer1_best.json").write_bytes(profile_path.read_bytes())
    write_json(run_dir / "selected_config.json", profile)
    write_json(profile_path, profile)
    yaml_path.write_text(candidate)
    print(f"PROMOTED validation-selected Layer 1: {profile_path}", flush=True)
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-floor", type=float, default=0.98)
    parser.add_argument("--reuse-encoder", action="store_true")
    parser.add_argument("--wait-for-encoder-pid", type=int,
                        help="Wait for an already-running encoder owned by this workflow before AE/calibration")
    parser.add_argument("--wait-timeout", type=int, default=7200)
    parser.add_argument("--promote", action="store_true")
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["CUDA_DEVICE_ML"] = args.device
    output = (args.run_dir or ROOT / "models/layer1/training_runs" / datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    yaml_hash = sha256(ROOT / "config/detection_config.yaml")
    if args.wait_for_encoder_pid:
        if not args.reuse_encoder:
            parser.error("--wait-for-encoder-pid requires --reuse-encoder")
        deadline = time.monotonic()+args.wait_timeout
        while not (output / "encoder/training_config.json").exists():
            write_json(output / "pipeline_status.json", {"phase": "waiting_for_encoder", "pid": args.wait_for_encoder_pid})
            if time.monotonic() > deadline:
                raise TimeoutError("Encoder wait deadline exceeded; no model promoted")
            try:
                os.kill(args.wait_for_encoder_pid, 0)
            except ProcessLookupError as exc:
                write_json(output / "pipeline_status.json", {"phase": "failed", "reason": "encoder exited without completed checkpoint"})
                raise RuntimeError("Encoder exited before writing completion metadata") from exc
            time.sleep(10)
        # Let the successful child release CUDA memory before checking GPU idleness.
        time.sleep(10)
    gpu = gpu_preflight(args.device)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.device, CUDA_DEVICE_ML=args.device,
               CUDA_DEVICE_TRAIN=args.device, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_PROGRESS_BARS="1")
    write_json(output / "pipeline_config.json", {"device": gpu, "seed": args.seed,
        "recall_floor": args.recall_floor, "prepared_manifest_sha256": sha256(ROOT / "data/training/manifest.json"),
        "detection_config_before_sha256": yaml_hash, "test_policy": "only_after_validation_selection_frozen",
        "encoder_hyperparameters": "declared_config;_best_epoch_by_validation_f1",
        "ae_grid": {"learning_rate": [1e-3, 3e-4], "latent_dim": [32, 64]},
        "reuse_encoder": args.reuse_encoder})

    def child(module, arguments, log):
        command = [sys.executable, "-u", "-m", module, "--device", args.device] + arguments
        print("RUN", " ".join(command), "LOG", log, flush=True)
        write_json(output / "pipeline_status.json", {"phase": module, "log": str(output / log)})
        try:
            with (output / log).open("w") as handle:
                subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        except Exception as exc:
            write_json(output / "pipeline_status.json", {"phase": "failed", "component": module, "error": str(exc)})
            raise

    encoder = output / "encoder"
    if args.reuse_encoder:
        record = json.loads((encoder / "training_config.json").read_text())
        if record["prepared_manifest_sha256"] != sha256(ROOT / "data/training/manifest.json") or record["smoke_test"]:
            raise ValueError("Encoder reuse provenance mismatch")
    else:
        child("training.modernbert_finetune", ["--smoke", "--train_percent", "100", "--seed", str(args.seed),
                                              "--output-dir", str(output / "smoke")], "smoke.log")
        child("training.modernbert_finetune", ["--train_percent", "100", "--seed", str(args.seed),
                                              "--output-dir", str(encoder)], "encoder.log")
    child("training.autoencoder_training", ["--encoder", str(encoder), "--output-dir", str(output / "autoencoder"),
                                            "--seed", str(args.seed), "--recall-floor", str(args.recall_floor),
                                            "--search", "--evaluate-test"], "autoencoder.log")
    write_json(output / "pipeline_status.json", {"phase": "runtime_consistency_check"})
    try:
        verify_runtime(output)
        if args.promote:
            promote(output, yaml_hash)
    except Exception as exc:
        write_json(output / "pipeline_status.json", {"phase": "failed", "component": "verification_or_promotion", "error": str(exc)})
        raise
    write_json(output / "pipeline_status.json", {"phase": "complete", "promoted": args.promote})
    print(f"PIPELINE COMPLETE: {output}", flush=True)
    return output


if __name__ == "__main__":
    main()
