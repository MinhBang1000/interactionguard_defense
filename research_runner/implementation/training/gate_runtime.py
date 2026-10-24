from pathlib import Path
import json
import subprocess
from research_runner.implementation.data.data_preprocess import write_json

def verify_runtime(run_dir):
    import numpy as np
    import torch
    from research_runner.implementation.data.prepare_training import load_prepared
    from research_runner.implementation.detectors.gate import GateDetector

    run_dir = Path(run_dir)
    selection = json.loads((run_dir / "autoencoder/selection_frozen.json").read_text())
    best = selection["best"]
    rows = load_prepared("encoder", "val")
    indices = sorted({0, next(i for i, r in enumerate(rows) if r["label"] == 1),
                      max(range(len(rows)), key=lambda i: len(rows[i]["sequence"]))})
    expected = np.load(Path(best["checkpoint"]).parent / "validation_errors.npy")[indices]
    detector = GateDetector(str(run_dir / "encoder"), best["checkpoint"], "cuda:0", best["calibration"]["threshold"])
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

