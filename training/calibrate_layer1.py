"""Validation-only recalibration of the active Layer 1; never reads test data."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="0")
    parser.add_argument("--recall-floor", type=float, default=0.98)
    args = parser.parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["CUDA_DEVICE_ML"] = args.device
    from training.layer1_pipeline import gpu_preflight
    gpu_preflight(args.device)
    import yaml
    from data.data_preprocess import ROOT, sha256, write_json
    from data.prepare_training import load_prepared
    from models.detectors.layer1_detector import Layer1Detector
    from training.layer1_common import calibrate

    config = yaml.safe_load((ROOT / "config/detection_config.yaml").read_text())
    rows = load_prepared("encoder", "val")
    detector = Layer1Detector(device="cuda:0", **config["layers"]["layer1"]["params"])
    result = detector.predict([r["sequence"] for r in rows], batch_size=1)
    calibration = calibrate([r["label"] for r in rows], result["errors"], args.recall_floor)
    report = {"source": "prepared_noncorrelated_validation_only", "test_used": False,
              "sample_ids": [r["id"] for r in rows], "calibration": calibration,
              "encoder_path": str(detector.bert_model_path), "ae_path": str(detector.ae_model_path),
              "ae_sha256": sha256(detector.ae_model_path)}
    write_json(ROOT / "results/layer1_threshold_search.json", report)
    print(calibration)
    print("Saved validation report; active threshold unchanged. Full pipeline promotion activates a selected model pair.")
    return report


if __name__ == "__main__":
    main()
