"""
Std Testing -- Phase 2: evaluate all 9 thesis Table 6.4 modes
(config/test_modes.yaml modes 1, 2, 3, 10, 11, 12, 13, 14, 15) against each
Phase-1 trial's Layer 1 / Layer 2 checkpoints, with Layer 3 pinned to the O2
variant (openai/gpt-4.1-mini via OpenRouter) -- the same Layer 3
configuration that produced the reference numbers in
results/mode_*_results.json (verified: every one of those files has
"selected_layer3_variant": "layer3o2"). Layer 3 itself is not retrained; it
still contributes real latency variance across trials/runs, which is one of
the things this feature is meant to capture.

Run this phase AFTER Phase 1 (experiments/std_testing/train.py) has produced
trial checkpoints under models/std_testing/. This is a separate, later
invocation, not chained automatically after Phase 1.
"""
from __future__ import annotations

import copy
import csv
import glob
import os
import shutil
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import yaml

from experiments.common import ask_with_default, load_detection_config, read_json, write_json, yes_no
from experiments.std_testing.common import (
    REFERENCE_LAYER1_THRESHOLD,
    REFERENCE_LAYER2_THRESHOLD,
    STD_TESTING_MODEL_DIR,
    STD_TESTING_RESULT_DIR,
    TABLE_6_4_MODES,
    trial_config_path,
    trial_model_dir,
    trial_name,
    trial_result_dir,
)
from utils.paths import PROCESSED_DIR
from utils.utils import load_jsonl, print_results, save_results

METRIC_KEYS = ["precision", "recall", "f1_score", "fpr", "fnr", "accuracy"]

# Modes that call the Layer 3 LLM (via OpenRouter) and therefore consume API
# quota. Modes 1, 2, 10, 15 only use Layer 1/2 and are free/instant.
LLM_MODES = {3, 11, 12, 13, 14}

# Minimum OpenRouter quota (USD) required before starting another LLM mode.
# Below this, Phase 2 stops cleanly so the user can swap the API key and
# resume later -- already-saved results are never touched.
MIN_QUOTA_USD = 10.0


def _check_openrouter_quota() -> Optional[float]:
    """Return remaining OpenRouter quota in USD, or None if the check itself
    could not be completed (missing key, network error, unexpected response).
    A None result is treated as "unknown" and does not block the run -- only
    a confirmed low balance does.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        if response.status_code != 200:
            return None
        remaining = response.json().get("data", {}).get("limit_remaining")
        return float(remaining) if remaining is not None else None
    except Exception:
        return None


def _discover_trials() -> List[Dict[str, Any]]:
    trials = []
    for manifest_path in sorted(glob.glob(str(STD_TESTING_MODEL_DIR / "*" / "manifest.json"))):
        trials.append(read_json(Path(manifest_path)))
    return trials


def _preflight_check(config: Dict[str, Any]) -> bool:
    layer1_threshold = config.get("layers", {}).get("layer1", {}).get("params", {}).get("threshold")
    layer2_threshold = config.get("layers", {}).get("layer2", {}).get("params", {}).get("threshold")
    layer3o2_params = config.get("layers", {}).get("layer3o2", {}).get("params", {})

    print("\n" + "=" * 72)
    print("PRE-FLIGHT CONFIG CHECK (config/detection_config.yaml)")
    print("=" * 72)
    print(f"Layer 1 threshold : {layer1_threshold}  (reference: {REFERENCE_LAYER1_THRESHOLD})")
    print(f"Layer 2 threshold : {layer2_threshold}  (reference: {REFERENCE_LAYER2_THRESHOLD})")
    print("Layer 3 O2 params (forced for every mode in this run via layer3_variant='o2'):")
    for key in ("model", "backend", "base_url", "threshold", "temperature", "max_tokens"):
        print(f"  {key}: {layer3o2_params.get(key)}")

    mismatch = False
    if layer1_threshold != REFERENCE_LAYER1_THRESHOLD:
        print(
            f"\n[WARNING] Layer 1 threshold {layer1_threshold} does not match the "
            f"reference value {REFERENCE_LAYER1_THRESHOLD} that reproduced Table 6.4."
        )
        mismatch = True
    if layer2_threshold != REFERENCE_LAYER2_THRESHOLD:
        print(
            f"[WARNING] Layer 2 threshold {layer2_threshold} does not match the "
            f"reference value {REFERENCE_LAYER2_THRESHOLD} that reproduced Table 6.4."
        )
        mismatch = True

    if mismatch:
        print(
            "\nThis run will NOT be comparable to the existing results/mode_*_results.json\n"
            "reference numbers. Fix config/detection_config.yaml thresholds first, or\n"
            "confirm you intend to establish a new reference point."
        )
        return yes_no("Continue anyway?", default=False)

    print("\nConfig matches the known reference values. OK to proceed.")
    return True


def _build_trial_detection_config(base_config: Dict[str, Any], trial: Dict[str, Any]) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_config)
    cfg["layers"]["layer1"]["params"]["bert_model_path"] = trial["modernbert_dir"]
    cfg["layers"]["layer1"]["params"]["ae_model_path"] = str(Path(trial["autoencoder_dir"]) / "autoencoder.pth")
    cfg["layers"]["layer2"]["params"]["model_path"] = trial["deberta_dir"]
    return cfg


def _write_trial_yaml(cfg: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_trial_modes(trial: Dict[str, Any], modes: List[int], overwrite: bool) -> bool:
    """Run the requested modes for one trial.

    Returns True if every requested mode ran (or was already present), or
    False if Phase 2 was stopped early because OpenRouter quota dropped
    below MIN_QUOTA_USD -- the caller should stop the whole run (not just
    this trial) in that case, since quota is shared across all trials.
    """
    from models.detectors.pipeline import DetectionPipeline

    seed = trial["seed"]
    index = trial["trial_index"]
    name = trial_name(seed, index)
    yaml_path = trial_config_path(seed, index)
    result_dir = trial_result_dir(seed, index)

    base_config = load_detection_config()
    trial_cfg = _build_trial_detection_config(base_config, trial)
    _write_trial_yaml(trial_cfg, yaml_path)

    test_data = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")
    if not test_data:
        raise ValueError("Empty test_processed.jsonl")
    sequences = [d["sequence"] for d in test_data]
    labels = np.array([d["label"] for d in test_data])

    invalidated_modes = {int(m) for m in base_config.get("std_testing_invalidated_modes", [])}

    for mode_id in modes:
        out_path = result_dir / f"mode_{mode_id}_results.json"
        is_invalidated = mode_id in invalidated_modes
        if out_path.exists() and not overwrite and not is_invalidated:
            print(f"  [skip] {name} mode {mode_id}: result already exists")
            continue
        if out_path.exists() and is_invalidated:
            print(
                f"  [rerun] {name} mode {mode_id}: marked stale in "
                f"detection_config.yaml (std_testing_invalidated_modes) -- "
                f"re-running and overwriting the old result."
            )

        if mode_id in LLM_MODES:
            remaining = _check_openrouter_quota()
            if remaining is None:
                print("  [warn] Could not check OpenRouter quota (network/key issue) -- continuing anyway.")
            elif remaining < MIN_QUOTA_USD:
                print(
                    f"\n[STOP] OpenRouter quota is ${remaining:.2f}, below the "
                    f"${MIN_QUOTA_USD:.0f} safety floor.\n"
                    f"Stopping before {name} | mode {mode_id} to avoid a run "
                    f"failing/corrupting mid-way through.\n"
                    f"All results saved so far are untouched. Swap "
                    f"OPENROUTER_API_KEY in .env, then re-run Phase 2 -- "
                    f"modes already completed will be skipped automatically."
                )
                return False
            else:
                print(f"  [quota] OpenRouter remaining: ${remaining:.2f} (ok, >= ${MIN_QUOTA_USD:.0f})")

        print(f"\n--- {name} | mode {mode_id} ---")
        pipeline = DetectionPipeline(
            config_path=yaml_path,
            test_mode=mode_id,
            layer3_variant="o2",
        )
        results = pipeline.predict(sequences=sequences, labels=labels, return_details=True)
        print_results(results)
        save_results(results, out_path)

    return True


def _trial_fully_evaluated(trial: Dict[str, Any]) -> bool:
    """True only if every Table 6.4 mode has a saved result for this trial.

    Used to gate checkpoint deletion: a trial's models/std_testing/ weights
    are only safe to delete once nothing in this feature will ever need to
    read them again (i.e. all 9 modes have already produced a result file).
    Deleting after a partial run (e.g. only the cheap non-LLM modes) would
    make the remaining modes impossible to run later.
    """
    result_dir = trial_result_dir(trial["seed"], trial["trial_index"])
    return all((result_dir / f"mode_{m}_results.json").exists() for m in TABLE_6_4_MODES)


def _maybe_cleanup_trial_checkpoints(trials: List[Dict[str, Any]]) -> None:
    fully_done = [t for t in trials if _trial_fully_evaluated(t)]
    if not fully_done:
        return

    print(f"\n{len(fully_done)} trial(s) have results for all 9 Table 6.4 modes:")
    for t in fully_done:
        model_dir = trial_model_dir(t["seed"], t["trial_index"])
        size_note = " (already gone)" if not model_dir.exists() else ""
        print(f"  - seed {t['seed']} -> {model_dir}{size_note}")

    print(
        "\nOnce a trial's results are saved, its models/std_testing/ checkpoint\n"
        "(ModernBERT + AutoEncoder + DeBERTa, ~5GB per trial) is no longer needed\n"
        "by this feature -- deleting it frees disk space for the remaining trials.\n"
        "This is IRREVERSIBLE: if you ever need to re-run a mode for that trial,\n"
        "you would have to retrain it from Phase 1 again."
    )
    if not yes_no("Delete checkpoints for the fully-evaluated trial(s) above?", default=False):
        return

    for t in fully_done:
        model_dir = trial_model_dir(t["seed"], t["trial_index"])
        if model_dir.exists():
            shutil.rmtree(model_dir)
            print(f"  Deleted: {model_dir}")


def _aggregate(trials: List[Dict[str, Any]], modes: List[int]) -> None:
    rows = []
    summary: Dict[str, Any] = {}

    for mode_id in modes:
        per_trial_metrics = []
        per_trial_latency = []
        for trial in trials:
            result_path = trial_result_dir(trial["seed"], trial["trial_index"]) / f"mode_{mode_id}_results.json"
            if not result_path.exists():
                continue
            data = read_json(result_path)
            per_trial_metrics.append(data.get("metrics", {}))
            num_samples = data.get("num_samples") or 1
            per_trial_latency.append(data.get("time", 0.0) / num_samples)

        if not per_trial_metrics:
            continue

        mode_summary: Dict[str, Any] = {"num_trials": len(per_trial_metrics)}
        row: Dict[str, Any] = {"mode": mode_id, "num_trials": len(per_trial_metrics)}

        for key in METRIC_KEYS:
            values = [m.get(key) for m in per_trial_metrics if m.get(key) is not None]
            if not values:
                continue
            mean = statistics.mean(values)
            std = statistics.stdev(values) if len(values) > 1 else 0.0
            mode_summary[key] = {"mean": mean, "std": std, "values": values}
            row[f"{key}_mean"] = round(mean, 4)
            row[f"{key}_std"] = round(std, 4)

        if per_trial_latency:
            mean_lat = statistics.mean(per_trial_latency)
            std_lat = statistics.stdev(per_trial_latency) if len(per_trial_latency) > 1 else 0.0
            mode_summary["latency_per_sample"] = {"mean": mean_lat, "std": std_lat, "values": per_trial_latency}
            row["latency_per_sample_mean"] = round(mean_lat, 4)
            row["latency_per_sample_std"] = round(std_lat, 4)

        summary[str(mode_id)] = mode_summary
        rows.append(row)

    write_json(STD_TESTING_RESULT_DIR / "summary.json", summary)
    print(f"Saved: {STD_TESTING_RESULT_DIR / 'summary.json'}")

    if rows:
        csv_path = STD_TESTING_RESULT_DIR / "summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {csv_path}")


def run_std_testing_aggregate() -> None:
    """Standalone aggregation step: recompute results/std_testing/summary.json
    and summary.csv from whatever per-trial mode result files already exist
    on disk, without running (or re-running) any evaluation.

    Useful after a partial Phase 2 run, after fixing a bug and re-running
    only some modes, or simply to refresh the summary on demand -- this does
    not call the LLM, does not touch GPU, and does not cost any OpenRouter
    quota.
    """
    print("\n" + "=" * 72)
    print("STD TESTING -- AGGREGATE: recompute summary.json / summary.csv")
    print("=" * 72)

    trials = _discover_trials()
    if not trials:
        print(f"No Phase 1 trials found under {STD_TESTING_MODEL_DIR}. Run Phase 1 first.")
        return
    print(f"Found {len(trials)} trial(s): seeds {[t['seed'] for t in trials]}")

    modes_raw = ask_with_default(
        f"Comma-separated mode IDs to aggregate [default: {','.join(str(m) for m in TABLE_6_4_MODES)}]: ",
        ",".join(str(m) for m in TABLE_6_4_MODES),
    )
    modes = [int(part.strip()) for part in modes_raw.split(",") if part.strip()]

    for mode_id in modes:
        found = sum(
            1 for t in trials
            if (trial_result_dir(t["seed"], t["trial_index"]) / f"mode_{mode_id}_results.json").exists()
        )
        print(f"  mode {mode_id}: {found}/{len(trials)} trial(s) have a saved result")

    _aggregate(trials, modes)
    print("\nAggregation complete.")


def run_std_testing_evaluate() -> None:
    print("\n" + "=" * 72)
    print("STD TESTING -- PHASE 2: EVALUATE ALL TABLE 6.4 MODES PER TRIAL")
    print("=" * 72)

    trials = _discover_trials()
    if not trials:
        print(f"No Phase 1 trials found under {STD_TESTING_MODEL_DIR}. Run Phase 1 first.")
        return

    print(f"Found {len(trials)} trial(s): seeds {[t['seed'] for t in trials]}")

    starting_quota = _check_openrouter_quota()
    if starting_quota is not None:
        print(f"OpenRouter quota remaining right now: ${starting_quota:.2f} (safety floor: ${MIN_QUOTA_USD:.0f})")
    else:
        print("Could not check OpenRouter quota right now (will keep trying before each LLM mode).")

    base_config = load_detection_config()
    if not _preflight_check(base_config):
        print("Aborted.")
        return

    invalidated_modes = sorted(int(m) for m in base_config.get("std_testing_invalidated_modes", []))
    if invalidated_modes:
        print(
            f"\nNote: modes {invalidated_modes} are marked stale in "
            f"config/detection_config.yaml (std_testing_invalidated_modes) and "
            f"will be re-run + overwritten for every trial automatically, even "
            f"though results already exist for them. All other modes keep the "
            f"normal skip-if-already-done behavior."
        )

    modes_raw = ask_with_default(
        f"Comma-separated mode IDs to run [default: {','.join(str(m) for m in TABLE_6_4_MODES)}]: ",
        ",".join(str(m) for m in TABLE_6_4_MODES),
    )
    modes = [int(part.strip()) for part in modes_raw.split(",") if part.strip()]

    print(
        "\nCost warning: modes 3, 11, 12, 13, 14 call the Layer 3 LLM\n"
        "(gpt-4.1-mini via OpenRouter) for every sample. Mode 3 alone took\n"
        "~4.5h in the original single run -- running it for every trial can\n"
        "take a long time. Existing per-trial/per-mode results are skipped\n"
        "automatically unless you choose to overwrite, so this can be run\n"
        "incrementally (e.g. one trial or one mode at a time).\n"
    )
    overwrite = yes_no("Overwrite existing per-trial results?", default=False)
    if not yes_no("Proceed?", default=True):
        print("Aborted.")
        return

    stopped_early = False
    for trial in trials:
        completed = _run_trial_modes(trial, modes, overwrite)
        if not completed:
            stopped_early = True
            break

    _aggregate(trials, modes)

    # Gated internally on actual result files on disk (not just this run's
    # `modes` list), so this is safe to call even after a partial run --
    # a trial only qualifies once every one of the 9 Table 6.4 modes has a
    # saved result, whether that happened in this invocation or an earlier one.
    _maybe_cleanup_trial_checkpoints(trials)

    if stopped_early:
        print("\nPhase 2 stopped early due to low OpenRouter quota (see [STOP] message above).")
    else:
        print("\nPhase 2 complete.")


if __name__ == "__main__":
    run_std_testing_evaluate()
