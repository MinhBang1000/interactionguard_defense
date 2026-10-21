"""Inference-only template check: pre-trained L1/L2 (5 Std Testing seeds) and RAGuard on original vs.
normalized data. No training, no LLM, no API. Outputs go to results/template_check/ only.

Configs (same routing rules as models/detectors/pipeline.py):
  L1    (mode 1)  : pred = L1                      (L1 Check counts as malicious)
  L2    (mode 2)  : pred = L2
  L1-L2 (mode 10) : pred = L1 AND L2               (unresolved L2 Check -> forwarded/benign)
  L2-L1 (mode 15) : pred = L2 OR L1                (L2 Stop, else L1 Check -> malicious)
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "template_check"
SEEDS = [(1, 42), (2, 43), (3, 44), (4, 45), (5, 46)]
CONFIG_MODE = {"L1": 1, "L2": 2, "L1-L2": 10, "L2-L1": 15}


def metrics(y, p):
    y = np.asarray(y); p = np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum()); fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum()); tn = int(((y == 0) & (p == 0)).sum())
    P = tp / (tp + fp) if tp + fp else float("nan"); R = tp / (tp + fn) if tp + fn else float("nan")
    F = 2 * P * R / (P + R) if (tp + fp) and (tp + fn) and (P + R) else float("nan")
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, P=P, R=R, F1=F, FPR=fp / (fp + tn) if fp + tn else float("nan"))


def summarize(labels, cats, pred):
    out = {"ALL": metrics(labels, pred)}
    by = defaultdict(list)
    for i, c in enumerate(cats):
        by[c].append(i)
    for c, idx in by.items():
        out[c] = metrics(labels[idx], pred[idx])
    return out


def run_l1_l2():
    from models.detectors.layer1_detector import Layer1Detector
    from models.detectors.layer2_detector import Layer2Detector

    datasets = {
        "original": [json.loads(l) for l in open(ROOT / "data/processed/test_processed.jsonl")],
        "normalized": [json.loads(l) for l in open(OUT / "test_processed_normalized.jsonl")],
    }
    labels = np.array([r["label"] for r in datasets["original"]])
    cats = [r["attack_type"] or "benign" for r in datasets["original"]]
    results = {}
    for idx, seed in SEEDS:
        name = f"trial{idx}_seed{seed}"
        cfg = yaml.safe_load(open(ROOT / f"config/std_testing/{name}_detection_config.yaml"))
        dev = f"cuda:{cfg['device']}" if str(cfg["device"]).isdigit() else str(cfg["device"])
        dev = os.environ.get("TEMPLATE_CHECK_DEVICE", dev)  # GPU choice does not change the model
        l1p = cfg["layers"]["layer1"]["params"]; l2p = cfg["layers"]["layer2"]["params"]
        assert float(l1p["threshold"]) == 0.05 and float(l2p["threshold"]) == 0.5, (l1p, l2p)
        l1 = Layer1Detector(device=dev, **l1p); l2 = Layer2Detector(device=dev, **l2p)
        results[name] = {}
        for dname, rows in datasets.items():
            seqs = [r["sequence"] for r in rows]
            emb = l1.extract_embedding(seqs, batch_size=128, cache_dir=str(OUT / "emb_cache" / f"{name}_{dname}"))
            p1 = (l1.compute_reconstruction_errors(emb) > l1.threshold).astype(int)
            p2 = np.asarray(l2.predict(seqs, batch_size=8)["predictions"]).astype(int)
            preds = {"L1": p1, "L2": p2, "L1-L2": p1 & p2, "L2-L1": p2 | p1}
            res = {k: summarize(labels, cats, v) for k, v in preds.items()}
            if dname == "original":  # reproduction check against the stored Std Testing results
                res["_repro"] = {}
                for k, mode in CONFIG_MODE.items():
                    stored = json.load(open(ROOT / f"results/std_testing/{name}/mode_{mode}_results.json"))
                    res["_repro"][k] = bool(np.array_equal(np.asarray(stored["predictions"]).astype(int), preds[k]))
            results[name][dname] = res
            print(name, dname, {k: round(v["ALL"]["F1"], 4) for k, v in res.items() if not k.startswith("_")},
                  res.get("_repro", ""), flush=True)
        del l1, l2
        import torch; torch.cuda.empty_cache()
    return results


def run_raguard():
    from models.baseline.raguard import RAGuardDetector

    nd = json.load(open(ROOT / "eval/eval_asr_without_defense.json"))
    gt_success = {k: set(v["success_ids"]) for k, v in nd.items()}
    det = RAGuardDetector(verbose=False, device=os.environ.get("TEMPLATE_CHECK_DEVICE"))
    det.load_thresholds(str(ROOT / "models/baseline/raguard_thresholds.json"))
    data = {
        "original": [json.loads(l) for l in open(ROOT / "models/baseline/raguard_predictions.jsonl")],
        "normalized": [json.loads(l) for l in open(OUT / "baseline_inputs_normalized.jsonl")],
    }
    stored = {r["id"]: int(r["pred"]) for r in data["original"]}
    out = {}
    for dname, rows in data.items():
        preds = det.predict(rows, return_details=False)
        p = {r["id"]: int(r["pred"]) for r in preds}
        y = [int(r["label"]) for r in rows]; pv = [p[r["id"]] for r in rows]
        cats = [r.get("attack_type") or "benign" for r in rows]
        asr = {}
        for c in ["prompt", "rag", "tool", "correlated"]:
            ids = [r["id"] for r in rows if (r.get("attack_type") or "benign") == c]
            succ = sum(1 for i in ids if p[i] == 0 and i in gt_success[c])
            asr[c] = {"success": succ, "total": len(ids), "asr": succ / len(ids)}
        out[dname] = {"episode_metrics": summarize(np.array(y), cats, np.array(pv)), "asr": asr}
        if dname == "original":
            out[dname]["matches_stored_predictions"] = all(p[i] == stored[i] for i in stored)
        print("raguard", dname, round(out[dname]["episode_metrics"]["ALL"]["F1"], 4),
              {c: round(a["asr"], 4) for c, a in asr.items()}, out[dname].get("matches_stored_predictions", ""), flush=True)
    return out


def main():
    res = {"l1_l2": run_l1_l2(), "raguard": run_raguard()}
    (OUT / "infer_results.json").write_text(json.dumps(res, indent=2))
    print("saved", OUT / "infer_results.json")


if __name__ == "__main__":
    main()
