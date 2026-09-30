from __future__ import annotations

import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config.config import Issue2HighBenignSweepConfig
from experiments.common import (
    ask_choice as _ask_choice,
    ask_with_default as _ask_with_default,
    layer3_variant_suffix as _layer3_variant_suffix,
    normalize_layer3_variant as _normalize_layer3_variant,
    read_json as _read_json,
    yes_no as _yes_no,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from utils.paths import CONFIG_DIR, PROCESSED_DIR, RESULT_DIR
from utils.utils import load_jsonl, save_results


def _ask_positive_int(prompt: str, default: str) -> int:
    while True:
        raw = _ask_with_default(prompt, default)
        try:
            value = int(raw)
        except ValueError:
            print("Invalid input. Please enter a positive integer.")
            continue
        if value >= 1:
            return value
        print("Invalid input. Please enter a positive integer.")


def _ask_pct(prompt: str, default: str) -> int:
    while True:
        raw = _ask_with_default(prompt, default)
        try:
            value = int(raw)
        except ValueError:
            print("Invalid input. Please enter an integer percentage from 1 to 99.")
            continue
        if 1 <= value <= 99:
            return value
        print("Invalid input. Please enter an integer percentage from 1 to 99.")


def _ask_layer3_variant() -> Optional[str]:
    print("\nLayer 3 version")
    print("This staged high-benign backend sweep uses Mode 11.")
    print("Press Enter for default/current logical layer3.\n")
    print("default) current logical layer3")
    print("orig)    Original Layer3Detector")
    print("o2)      Layer3DetectorO2")
    print("custom configured key such as layer3o2")

    with open(CONFIG_DIR / "detection_config.yaml", "r", encoding="utf-8") as f:
        import yaml

        detection_cfg = yaml.safe_load(f) or {}
    configured_layers = set((detection_cfg.get("layers") or {}).keys())

    while True:
        raw = input("Select Layer 3 version [default]: ").strip().lower()
        if raw in {"", "default"}:
            print("Using default logical layer3.\n")
            return None
        if raw in {"o2", "layer3o2"}:
            variant = "layer3o2"
        elif raw in {"orig", "original", "layer3orig"}:
            variant = "layer3orig"
        elif raw.startswith("layer3"):
            variant = raw
        else:
            print("Invalid choice. Please choose default, orig, o2, or layer3*.")
            continue

        if variant not in configured_layers:
            print(f"{variant} is not configured in detection_config.yaml.")
            continue

        print(f"Using {variant} for logical layer3.\n")
        return variant


def _current_dir(layer3_variant: Optional[str]) -> Path:
    suffix = _layer3_variant_suffix(layer3_variant)
    return RESULT_DIR / "backend_latency_sweep_high_benign" / f"mode_11{suffix}" / "current"


def _load_layer1_threshold_from_detection_config() -> float:
    with open(CONFIG_DIR / "detection_config.yaml", "r", encoding="utf-8") as f:
        import yaml

        config_data = yaml.safe_load(f) or {}
    return float(
        config_data.get("layers", {})
        .get("layer1", {})
        .get("params", {})
        .get("threshold", 0.14)
    )


def _sample_high_benign_subset(
    test_data: List[Dict[str, Any]],
    total_size: int,
    benign_ratio: float,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    benign_indices = [i for i, d in enumerate(test_data) if d["label"] == 0]
    malicious_indices_by_type = {
        atk: [
            i
            for i, d in enumerate(test_data)
            if d["label"] == 1 and d.get("attack_type") == atk
        ]
        for atk in Issue2HighBenignSweepConfig.ATTACK_TYPES
    }

    n_benign = int(total_size * benign_ratio)
    n_mal_total = total_size - n_benign

    if n_benign > len(benign_indices):
        raise ValueError(
            f"Need {n_benign} benign samples for ratio {benign_ratio:.2f}, "
            f"but only {len(benign_indices)} are available."
        )

    selected_benign = rng.sample(benign_indices, n_benign)

    n_attack_types = len(Issue2HighBenignSweepConfig.ATTACK_TYPES)
    balanced_take = n_mal_total // n_attack_types
    selected_malicious: List[int] = []
    remaining_by_type: Dict[str, List[int]] = {}

    for attack_type in Issue2HighBenignSweepConfig.ATTACK_TYPES:
        pool = malicious_indices_by_type[attack_type]
        take = min(balanced_take, len(pool))
        chosen = rng.sample(pool, take)
        selected_malicious.extend(chosen)
        chosen_set = set(chosen)
        remaining_by_type[attack_type] = [idx for idx in pool if idx not in chosen_set]

    remaining_needed = n_mal_total - len(selected_malicious)
    if remaining_needed > 0:
        remaining_pool: List[int] = []
        for attack_type in Issue2HighBenignSweepConfig.ATTACK_TYPES:
            remaining_pool.extend(remaining_by_type[attack_type])

        if remaining_needed > len(remaining_pool):
            raise ValueError(
                f"Need {remaining_needed} extra malicious samples after balanced sampling, "
                f"but only {len(remaining_pool)} remain."
            )

        selected_malicious.extend(rng.sample(remaining_pool, remaining_needed))

    sampled_indices = selected_benign + selected_malicious
    rng.shuffle(sampled_indices)
    sampled = [test_data[i] for i in sampled_indices]

    per_attack_counts = {}
    for attack_type in Issue2HighBenignSweepConfig.ATTACK_TYPES:
        per_attack_counts[attack_type] = sum(
            1
            for i in selected_malicious
            if test_data[i].get("attack_type") == attack_type
        )

    return sampled, {
        "benign_count": n_benign,
        "malicious_count": n_mal_total,
        "actual_benign_ratio": n_benign / total_size,
        "attack_type_counts": per_attack_counts,
    }


def _run_mode11_until_layer3(pipeline: Any, sequences: List[str], labels: np.ndarray) -> Dict[str, Any]:
    n = len(sequences)
    scores = np.zeros(n, dtype=float)
    final_before_l3 = np.zeros(n, dtype=int)
    layer_stats: Dict[str, Any] = {}

    stage_start = time.perf_counter()

    t1 = time.perf_counter()
    out1 = pipeline._run_layer("layer1", sequences, batch_size=128)
    layer1_elapsed = time.perf_counter() - t1
    p1 = out1["predictions"]
    s1 = out1["probabilities"]
    scores[:] = s1
    idx_susp = np.where(p1 == 1)[0]

    layer_stats["layer1"] = {
        "processed": n,
        "forwarded": int((p1 == 0).sum()),
        "sent_to_next": int((p1 == 1).sum()),
        "time": float(layer1_elapsed),
    }
    pipeline._attach_layer_confusion(layer_stats, "layer1", labels, p1)

    idx_to_l3 = np.array([], dtype=int)
    layer2_elapsed = 0.0
    if len(idx_susp) > 0:
        sub2 = [sequences[i] for i in idx_susp]
        t2 = time.perf_counter()
        out2 = pipeline._run_layer("layer2", sub2, batch_size=4)
        layer2_elapsed = time.perf_counter() - t2
        p2 = out2["predictions"]
        s2 = out2["probabilities"]
        scores[idx_susp] = s2

        idx_block = idx_susp[p2 == 1]
        final_before_l3[idx_block] = 1
        idx_to_l3 = idx_susp[p2 == 0]

        layer_stats["layer2"] = {
            "processed": int(len(idx_susp)),
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
            "time": float(layer2_elapsed),
        }
        pipeline._attach_layer_confusion(layer_stats, "layer2", labels[idx_susp], p2)
    else:
        layer_stats["layer2"] = {
            "processed": 0,
            "blocked": 0,
            "sent_to_next": 0,
            "time": 0.0,
        }

    return {
        "scores": scores,
        "final_before_l3": final_before_l3,
        "idx_to_l3": idx_to_l3,
        "layer_stats": layer_stats,
        "l1_l2_wall_time_seconds": float(time.perf_counter() - stage_start),
        "layer1_time_seconds": float(layer1_elapsed),
        "layer2_time_seconds": float(layer2_elapsed),
        "layer1_embedding_cache_loaded": bool(out1["raw"].get("used_cached_embeddings", False)),
        "layer1_embedding_batch_size": int(out1["raw"].get("embedding_batch_size", 128)),
    }


def _call_layer3_for_latency(layer3: Any, sequence: str) -> Tuple[int, float, Dict[str, Any]]:
    if hasattr(layer3, "_call_llm"):
        result = layer3._call_llm(str(sequence or "").strip())
        confidence = float(result.get("confidence", 0.0) or 0.0)
        threshold = float(getattr(layer3, "threshold", 0.5))
        is_attack = bool(result.get("is_malicious", False)) and confidence >= threshold
        return int(is_attack), confidence, result

    detected, score = layer3.detect_single(str(sequence or "").strip())
    return int(bool(detected)), float(score), {"reason": "detect_single"}


def _replay_layer3_backend(
    layer3: Any,
    sequences: List[str],
    labels: np.ndarray,
    sample_indices: np.ndarray,
    agent_concurrency: int,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    predictions_by_agent = [
        np.zeros(len(sequences), dtype=int)
        for _ in range(int(agent_concurrency))
    ]
    probabilities_by_agent = [
        np.zeros(len(sequences), dtype=float)
        for _ in range(int(agent_concurrency))
    ]
    details_by_agent: List[List[Optional[Dict[str, Any]]]] = [
        [None] * len(sequences)
        for _ in range(int(agent_concurrency))
    ]
    agent_streams: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()

    def agent_stream(agent_id: int) -> Tuple[int, List[Dict[str, Any]], Dict[str, Any]]:
        stream_events: List[Dict[str, Any]] = []
        stream_start = time.perf_counter()
        for local_idx, sequence in enumerate(sequences):
            label = labels[local_idx] if labels is not None else None
            t_submit_wall = time.time()
            t_submit_perf = time.perf_counter()
            try:
                pred, score, detail = _call_layer3_for_latency(layer3, sequence)
                error = None
            except Exception as exc:
                pred = 0
                score = 0.0
                detail = {"reason": f"Layer3 replay error: {str(exc)[:120]}"}
                error = str(exc)[:240]
            t_receive_perf = time.perf_counter()

            stream_events.append({
                "agent_id": int(agent_id),
                "agent_concurrency": int(agent_concurrency),
                "local_index": int(local_idx),
                "sample_index": int(sample_indices[local_idx]),
                "label": int(label) if label is not None else None,
                "t_send_epoch": float(t_submit_wall),
                "t_receive_epoch": float(t_submit_wall + (t_receive_perf - t_submit_perf)),
                "observed_latency_seconds": float(t_receive_perf - t_submit_perf),
                "prediction": int(pred),
                "probability": float(score),
                "error": error,
                "detail": detail,
            })

        stream_elapsed = time.perf_counter() - stream_start
        stream_values = np.array(
            [float(event["observed_latency_seconds"]) for event in stream_events],
            dtype=float,
        )
        stream_summary = {
            "agent_id": int(agent_id),
            "call_count": int(len(stream_events)),
            "wall_time_seconds": float(stream_elapsed),
            "mean_observed_seconds": float(stream_values.mean()) if stream_values.size else 0.0,
            "p50_observed_seconds": float(np.percentile(stream_values, 50)) if stream_values.size else 0.0,
            "p95_observed_seconds": float(np.percentile(stream_values, 95)) if stream_values.size else 0.0,
        }
        return int(agent_id), stream_events, stream_summary

    with ThreadPoolExecutor(max_workers=int(agent_concurrency)) as executor:
        futures = [
            executor.submit(agent_stream, agent_id)
            for agent_id in range(int(agent_concurrency))
        ]
        for future in as_completed(futures):
            agent_id, stream_events, stream_summary = future.result()
            agent_streams.append(stream_summary)
            for event in stream_events:
                local_idx = int(event["local_index"])
                predictions_by_agent[agent_id][local_idx] = int(event["prediction"])
                probabilities_by_agent[agent_id][local_idx] = float(event["probability"])
                details_by_agent[agent_id][local_idx] = event["detail"]
            events.extend(stream_events)

    events.sort(key=lambda row: (row["agent_id"], row["local_index"]))
    agent_streams.sort(key=lambda row: row["agent_id"])
    return {
        "predictions": predictions_by_agent[0] if predictions_by_agent else np.array([], dtype=int),
        "probabilities": probabilities_by_agent[0] if probabilities_by_agent else np.array([], dtype=float),
        "details": details_by_agent[0] if details_by_agent else [],
        "predictions_by_agent": predictions_by_agent,
        "probabilities_by_agent": probabilities_by_agent,
        "agent_streams": agent_streams,
        "events": events,
        "wall_time_seconds": float(time.perf_counter() - wall_start),
    }


def _latency_stats(
    events: List[Dict[str, Any]],
    baseline_seconds: Optional[float] = None,
    baseline_by_local_index: Optional[Dict[int, float]] = None,
) -> Dict[str, Any]:
    values = np.array([float(event["observed_latency_seconds"]) for event in events], dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean_observed_seconds": 0.0,
            "p50_observed_seconds": 0.0,
            "p95_observed_seconds": 0.0,
            "max_observed_seconds": 0.0,
            "baseline_service_seconds": baseline_seconds,
            "mean_effective_backend_delay_seconds": None,
            "p50_effective_backend_delay_seconds": None,
            "p95_effective_backend_delay_seconds": None,
            "mean_paired_effective_backend_delay_seconds": None,
            "p50_paired_effective_backend_delay_seconds": None,
            "p95_paired_effective_backend_delay_seconds": None,
        }

    stats: Dict[str, Any] = {
        "count": int(values.size),
        "mean_observed_seconds": float(values.mean()),
        "p50_observed_seconds": float(np.percentile(values, 50)),
        "p95_observed_seconds": float(np.percentile(values, 95)),
        "max_observed_seconds": float(values.max()),
        "baseline_service_seconds": baseline_seconds,
        "mean_effective_backend_delay_seconds": None,
        "p50_effective_backend_delay_seconds": None,
        "p95_effective_backend_delay_seconds": None,
        "mean_paired_effective_backend_delay_seconds": None,
        "p50_paired_effective_backend_delay_seconds": None,
        "p95_paired_effective_backend_delay_seconds": None,
    }

    if baseline_seconds is not None:
        effective = np.maximum(values - float(baseline_seconds), 0.0)
        stats.update({
            "mean_effective_backend_delay_seconds": float(effective.mean()),
            "p50_effective_backend_delay_seconds": float(np.percentile(effective, 50)),
            "p95_effective_backend_delay_seconds": float(np.percentile(effective, 95)),
        })

    if baseline_by_local_index:
        paired = []
        for event in events:
            local_idx = int(event["local_index"])
            if local_idx not in baseline_by_local_index:
                continue
            paired.append(
                max(
                    0.0,
                    float(event["observed_latency_seconds"])
                    - float(baseline_by_local_index[local_idx]),
                )
            )
        if paired:
            paired_values = np.array(paired, dtype=float)
            stats.update({
                "mean_paired_effective_backend_delay_seconds": float(paired_values.mean()),
                "p50_paired_effective_backend_delay_seconds": float(np.percentile(paired_values, 50)),
                "p95_paired_effective_backend_delay_seconds": float(np.percentile(paired_values, 95)),
            })

    return stats


def _gd1_artifact_name(benign_pct: int, run_idx: int) -> str:
    return f"benign_{int(benign_pct)}_run{int(run_idx)}.json"


def _candidates_name(benign_pct: int, run_idx: int) -> str:
    return f"l3_candidates_benign_{int(benign_pct)}_run{int(run_idx)}.jsonl"


def _gd2_result_name(benign_pct: int, run_idx: int, agent_concurrency: int) -> str:
    return f"benign_{int(benign_pct)}_run{int(run_idx)}_agentc{int(agent_concurrency)}.json"


def _gd2_events_name(benign_pct: int, run_idx: int, agent_concurrency: int) -> str:
    return f"l3_events_benign_{int(benign_pct)}_run{int(run_idx)}_agentc{int(agent_concurrency)}.jsonl"


def _load_existing_baseline(
    gd2_dir: Path,
    benign_pct: int,
    run_idx: int,
) -> Tuple[Optional[float], Optional[Dict[int, float]]]:
    result_path = gd2_dir / _gd2_result_name(benign_pct, run_idx, 1)
    events_path = gd2_dir / _gd2_events_name(benign_pct, run_idx, 1)
    if not (result_path.exists() and events_path.exists()):
        return None, None

    result = _read_json(result_path)
    events = load_jsonl(events_path)
    baseline_seconds = (
        result.get("l3_backend_latency", {}).get("p50_observed_seconds")
        or result.get("sweep_metadata", {}).get("baseline_service_seconds")
    )
    if baseline_seconds is None:
        baseline_seconds = _latency_stats(events)["p50_observed_seconds"]

    baseline_by_local_index = {
        int(event["local_index"]): float(event["observed_latency_seconds"])
        for event in events
        if int(event.get("agent_id", 0)) == 0
    }
    return float(baseline_seconds), baseline_by_local_index


def _summary_row_from_result(
    result: Dict[str, Any],
    benign_pct: int,
    run_idx: int,
    agent_concurrency: int,
    result_file: str,
) -> Dict[str, Any]:
    metrics = result.get("metrics", {}) or {}
    metadata = result.get("sweep_metadata", {}) or {}
    timing = result.get("timing_components", {}) or {}
    latency = result.get("l3_backend_latency", {}) or {}
    num_samples = int(result.get("num_samples", metadata.get("num_samples", 0)) or 0)
    pipeline_total = float(
        timing.get("pipeline_total_time_seconds", result.get("raw_observed_wall_time", result.get("time", 0.0))) or 0.0
    )

    return {
        "benign_pct": int(benign_pct),
        "ratio": float(metadata.get("actual_benign_ratio", benign_pct / 100.0) or 0.0),
        "run": int(run_idx),
        "backend_concurrency": int(agent_concurrency),
        "agent_concurrency": int(agent_concurrency),
        "layer3_variant": metadata.get("layer3_variant", result.get("selected_layer3_variant", "default")),
        "num_samples": num_samples,
        "benign_count": int(metadata.get("benign_count", 0) or 0),
        "malicious_count": int(metadata.get("malicious_count", 0) or 0),
        "l2_processed": int(metadata.get("l2_processed", 0) or 0),
        "l2_ratio": float(metadata.get("l2_ratio", 0.0) or 0.0),
        "l3_processed": int(metadata.get("l3_processed", metadata.get("l3_candidates", 0)) or 0),
        "l3_calls_total": int(metadata.get("l3_calls_total", 0) or 0),
        "l3_ratio": float(metadata.get("l3_ratio", 0.0) or 0.0),
        "l1_l2_wall_time_seconds": float(timing.get("l1_l2_wall_time_seconds", 0.0) or 0.0),
        "l3_backend_wall_time_seconds": float(
            timing.get("l3_backend_wall_time_seconds", latency.get("observed_backend_wall_time_seconds", 0.0)) or 0.0
        ),
        "pipeline_total_time_seconds": pipeline_total,
        "pipeline_latency_per_sample_seconds": float(
            timing.get(
                "pipeline_latency_per_sample_seconds",
                pipeline_total / num_samples if num_samples else 0.0,
            )
            or 0.0
        ),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("f1_score", metrics.get("f1", 0.0)) or 0.0),
        "fpr": float(metrics.get("fpr", 0.0) or 0.0),
        "fnr": float(metrics.get("fnr", 0.0) or 0.0),
        "baseline_source_concurrency": int(metadata.get("baseline_source_concurrency", 1) or 1),
        "result_file": result_file,
        "count": int(latency.get("count", 0) or 0),
        "mean_observed_seconds": float(latency.get("mean_observed_seconds", 0.0) or 0.0),
        "p50_observed_seconds": float(latency.get("p50_observed_seconds", 0.0) or 0.0),
        "p95_observed_seconds": float(latency.get("p95_observed_seconds", 0.0) or 0.0),
        "max_observed_seconds": float(latency.get("max_observed_seconds", 0.0) or 0.0),
        "baseline_service_seconds": latency.get("baseline_service_seconds"),
        "mean_effective_backend_delay_seconds": latency.get("mean_effective_backend_delay_seconds"),
        "p50_effective_backend_delay_seconds": latency.get("p50_effective_backend_delay_seconds"),
        "p95_effective_backend_delay_seconds": latency.get("p95_effective_backend_delay_seconds"),
        "mean_paired_effective_backend_delay_seconds": latency.get("mean_paired_effective_backend_delay_seconds"),
        "p50_paired_effective_backend_delay_seconds": latency.get("p50_paired_effective_backend_delay_seconds"),
        "p95_paired_effective_backend_delay_seconds": latency.get("p95_paired_effective_backend_delay_seconds"),
    }


def _write_gd2_summary(summary_rows: List[Dict[str, Any]], gd2_dir: Path, current_dir: Path) -> None:
    if not summary_rows:
        return
    summary = pd.DataFrame(summary_rows).sort_values(["benign_pct", "run", "agent_concurrency"])
    summary.to_csv(gd2_dir / "summary.csv", index=False)
    summary.to_csv(current_dir / "summary.csv", index=False)


def _run_gd1(
    layer3_variant: Optional[str],
    total_size: int,
    repeats: int,
    benign_start_pct: int,
    benign_end_pct: int,
    benign_step_pct: int,
    random_seed: int,
    overwrite: bool = False,
) -> Path:
    print("\n[GD_1] Running high-benign Mode 11 up to Layer 1 + Layer 2 only.")

    layer3_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _current_dir(layer3_variant)
    if current_dir.exists():
        if not overwrite:
            print("[GD_1] Existing current/ artifacts found; keeping them unchanged.")
            print(f"Output: {current_dir / 'gd1'}")
            return current_dir
        print("[GD_1] Overwriting existing current/ artifacts for this variant.\n")
        shutil.rmtree(current_dir)

    gd1_dir = current_dir / "gd1"
    gd1_dir.mkdir(parents=True, exist_ok=True)

    test_data = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")
    if not test_data:
        raise ValueError("Empty test-data")

    threshold = _load_layer1_threshold_from_detection_config()
    benign_available = sum(1 for d in test_data if d["label"] == 0)
    malicious_available = sum(1 for d in test_data if d["label"] == 1)
    benign_pcts = list(range(benign_start_pct, benign_end_pct + 1, benign_step_pct))

    manifest = {
        "stage": "GD_1",
        "sweep_type": "high_benign_backend_latency_staged",
        "mode_id": 11,
        "mode": "layer1_to_layer2_to_layer3",
        "layer3_variant": layer3_variant or "default",
        "fixed_total_size": int(total_size),
        "repeats": int(repeats),
        "benign_start_pct": int(benign_start_pct),
        "benign_end_pct": int(benign_end_pct),
        "benign_step_pct": int(benign_step_pct),
        "benign_pcts": benign_pcts,
        "random_seed": int(random_seed),
        "layer1_threshold_from_config": float(threshold),
        "available_counts": {
            "benign": int(benign_available),
            "malicious": int(malicious_available),
        },
        "sampling_policy": "balanced_across_attack_types_then_random_fill",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_contract": (
            "GD_1 freezes high-benign samples, L1/L2 routing, labels, scores, "
            "and timing. GD_2 replays only the saved L3 candidates."
        ),
    }
    _write_json(gd1_dir / "config.json", manifest)
    _write_json(current_dir / "sweep_config.json", manifest)

    from models.detectors.pipeline import DetectionPipeline

    summary_rows = []
    for benign_pct in benign_pcts:
        benign_ratio = benign_pct / 100.0
        print(f"\n[GD_1 benign ratio = {benign_pct}%]")

        for run_idx in range(1, repeats + 1):
            print(f"  -> Repeat {run_idx}/{repeats}")
            rng = random.Random(random_seed + benign_pct * 100 + run_idx)
            sampled, sample_meta = _sample_high_benign_subset(
                test_data=test_data,
                total_size=total_size,
                benign_ratio=benign_ratio,
                rng=rng,
            )
            sequences = [d["sequence"] for d in sampled]
            labels = np.array([d["label"] for d in sampled], dtype=int)

            print(
                "     "
                f"fixed_total_size={total_size}, "
                f"benign={sample_meta['benign_count']}, "
                f"malicious={sample_meta['malicious_count']}, "
                f"attack_mix={sample_meta['attack_type_counts']}"
            )

            pipeline = DetectionPipeline(
                config_path=CONFIG_DIR / "detection_config.yaml",
                test_mode=11,
                layer3_variant=layer3_variant,
            )
            if "layer1" in pipeline.layers:
                pipeline.layers["layer1"].threshold = float(threshold)

            partial = _run_mode11_until_layer3(pipeline, sequences, labels)
            idx_to_l3 = np.array(partial["idx_to_l3"], dtype=int)
            candidates_file = _candidates_name(benign_pct, run_idx)
            candidate_rows = []
            for local_idx, sample_idx in enumerate(idx_to_l3):
                source = sampled[int(sample_idx)]
                candidate_rows.append({
                    "local_index": int(local_idx),
                    "sample_index": int(sample_idx),
                    "label": int(labels[int(sample_idx)]),
                    "attack_type": str(source.get("attack_type", "")),
                    "sequence": source["sequence"],
                })
            _write_jsonl(gd1_dir / candidates_file, candidate_rows)

            layer1_stats = partial["layer_stats"].get("layer1", {})
            l2_processed = int(layer1_stats.get("sent_to_next", 0))
            l3_processed = int(len(idx_to_l3))
            l2_ratio = float(l2_processed / len(sequences)) if sequences else 0.0
            l3_ratio = float(l3_processed / len(sequences)) if sequences else 0.0
            l1_l2_time = float(partial["l1_l2_wall_time_seconds"])

            artifact = {
                "stage": "GD_1",
                "benign_pct": int(benign_pct),
                "ratio": float(sample_meta["actual_benign_ratio"]),
                "run": int(run_idx),
                "num_samples": int(len(sequences)),
                "labels": labels.tolist(),
                "final_before_l3": np.array(partial["final_before_l3"], dtype=int).tolist(),
                "scores_before_l3": np.array(partial["scores"], dtype=float).tolist(),
                "idx_to_l3": idx_to_l3.tolist(),
                "layer_stats": partial["layer_stats"],
                "layer1_embedding_cache_loaded": partial["layer1_embedding_cache_loaded"],
                "layer1_embedding_batch_size": partial["layer1_embedding_batch_size"],
                "layer_implementations": dict(pipeline.layer_aliases),
                "selected_layer3_variant": pipeline.layer_aliases.get("layer3", "layer3"),
                "candidates_file": candidates_file,
                "sample_metadata": sample_meta,
                "timing_components": {
                    "l1_l2_wall_time_seconds": l1_l2_time,
                    "layer1_time_seconds": float(partial["layer1_time_seconds"]),
                    "layer2_time_seconds": float(partial["layer2_time_seconds"]),
                    "l3_backend_wall_time_seconds": None,
                    "pipeline_total_time_seconds": None,
                    "pipeline_latency_per_sample_seconds": None,
                },
                "routing": {
                    "l2_processed": l2_processed,
                    "l2_ratio": l2_ratio,
                    "l3_processed": l3_processed,
                    "l3_ratio": l3_ratio,
                    "blocked_before_l3": int(np.array(partial["final_before_l3"], dtype=int).sum()),
                },
            }
            _write_json(gd1_dir / _gd1_artifact_name(benign_pct, run_idx), artifact)

            summary_rows.append({
                "benign_pct": int(benign_pct),
                "ratio": float(sample_meta["actual_benign_ratio"]),
                "run": int(run_idx),
                "layer3_variant": layer3_variant or "default",
                "fixed_total_size": int(total_size),
                "num_samples": int(len(sequences)),
                "benign_count": int(sample_meta["benign_count"]),
                "malicious_count": int(sample_meta["malicious_count"]),
                "l2_processed": l2_processed,
                "l2_ratio": l2_ratio,
                "l3_processed": l3_processed,
                "l3_ratio": l3_ratio,
                "blocked_before_l3": int(np.array(partial["final_before_l3"], dtype=int).sum()),
                "l1_l2_wall_time_seconds": l1_l2_time,
                "layer1_time_seconds": float(partial["layer1_time_seconds"]),
                "layer2_time_seconds": float(partial["layer2_time_seconds"]),
                "candidates_file": candidates_file,
                "gd1_artifact": _gd1_artifact_name(benign_pct, run_idx),
            })

    pd.DataFrame(summary_rows).sort_values(["benign_pct", "run"]).to_csv(gd1_dir / "summary.csv", index=False)
    pd.DataFrame(summary_rows).sort_values(["benign_pct", "run"]).to_csv(current_dir / "gd1_summary.csv", index=False)

    print("\nGD_1 completed.")
    print(f"Output: {gd1_dir}")
    return current_dir


def _run_gd2(
    layer3_variant: Optional[str],
    target_agent_concurrency: int,
    overwrite: bool = False,
) -> Path:
    print("\n[GD_2] Replaying saved high-benign Layer 3 candidates.")
    if overwrite:
        print("[GD_2] Existing GD_2 artifacts will be overwritten.\n")
    else:
        print("[GD_2] Existing completed artifacts will be reused and skipped.\n")

    layer3_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _current_dir(layer3_variant)
    gd1_dir = current_dir / "gd1"
    gd2_dir = current_dir / "gd2"

    if not (gd1_dir / "config.json").exists():
        raise FileNotFoundError(
            f"GD_1 artifacts not found: {gd1_dir}. Run GD_1 first for this Layer 3 variant."
        )
    if int(target_agent_concurrency) < 1:
        raise ValueError("Target Layer 3 agent concurrency must be >= 1")

    if overwrite and gd2_dir.exists():
        shutil.rmtree(gd2_dir)
    if overwrite and (current_dir / "summary.csv").exists():
        (current_dir / "summary.csv").unlink()
    gd2_dir.mkdir(parents=True, exist_ok=True)
    gd1_config = _read_json(gd1_dir / "config.json")
    layer3_variant = _normalize_layer3_variant(gd1_config.get("layer3_variant"))
    benign_pcts = [int(x) for x in gd1_config.get("benign_pcts", [])]
    repeats = int(gd1_config.get("repeats", 1))
    replay_concurrency_values = [1] if int(target_agent_concurrency) == 1 else [1, int(target_agent_concurrency)]

    gd2_config = {
        "stage": "GD_2",
        "sweep_type": "high_benign_backend_latency_staged",
        "mode_id": 11,
        "mode": "layer1_to_layer2_to_layer3",
        "layer3_variant": layer3_variant or "default",
        "baseline_agent_concurrency": 1,
        "target_agent_concurrency": int(target_agent_concurrency),
        "concurrency_values": replay_concurrency_values,
        "gd1_config": "gd1/config.json",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "measurement_note": (
            "GD_2 reuses frozen high-benign GD_1 L1/L2 routing. "
            "pipeline_total_time_seconds = GD_1 l1_l2_wall_time_seconds "
            "+ GD_2 l3_backend_wall_time_seconds."
        ),
    }
    _write_json(gd2_dir / "config.json", gd2_config)

    from models.detectors.pipeline import DetectionPipeline
    from utils.metrics import (
        compute_complete_metrics,
        compute_precision_recall_curve_data,
        compute_roc_curve_data,
    )

    pipeline = DetectionPipeline(
        config_path=CONFIG_DIR / "detection_config.yaml",
        test_mode=11,
        layer3_variant=layer3_variant,
    )

    summary_rows: List[Dict[str, Any]] = []
    for benign_pct in benign_pcts:
        print(f"\n[GD_2 benign ratio = {benign_pct}%]")
        for run_idx in range(1, repeats + 1):
            gd1_artifact = _read_json(gd1_dir / _gd1_artifact_name(benign_pct, run_idx))
            candidate_rows = load_jsonl(gd1_dir / gd1_artifact["candidates_file"])

            labels = np.array(gd1_artifact["labels"], dtype=int)
            final_before_l3 = np.array(gd1_artifact["final_before_l3"], dtype=int)
            scores_before_l3 = np.array(gd1_artifact["scores_before_l3"], dtype=float)
            idx_to_l3 = np.array(gd1_artifact["idx_to_l3"], dtype=int)
            l3_sequences = [row["sequence"] for row in candidate_rows]
            l3_labels = np.array([row["label"] for row in candidate_rows], dtype=int)
            sample_indices = np.array([row["sample_index"] for row in candidate_rows], dtype=int)

            baseline_seconds = None
            baseline_by_local_index = None

            for agent_concurrency in replay_concurrency_values:
                out_file = _gd2_result_name(benign_pct, run_idx, agent_concurrency)
                events_file = _gd2_events_name(benign_pct, run_idx, agent_concurrency)
                out_path = gd2_dir / out_file
                events_path = gd2_dir / events_file

                print(
                    f"  -> run={run_idx}, agent concurrency={agent_concurrency}, "
                    f"L3 candidates per agent={len(l3_sequences)}"
                )

                if out_path.exists() and events_path.exists():
                    print(f"     skip existing: {out_file}")
                    existing_result = _read_json(out_path)
                    if int(agent_concurrency) == 1 and baseline_seconds is None:
                        baseline_seconds, baseline_by_local_index = _load_existing_baseline(
                            gd2_dir,
                            benign_pct,
                            run_idx,
                        )
                    summary_rows.append(
                        _summary_row_from_result(
                            existing_result,
                            benign_pct,
                            run_idx,
                            agent_concurrency,
                            out_file,
                        )
                    )
                    _write_gd2_summary(summary_rows, gd2_dir, current_dir)
                    continue

                if out_path.exists() or events_path.exists():
                    print(f"     incomplete existing artifact found; rerunning: {out_file}")

                replay = _replay_layer3_backend(
                    pipeline.layers["layer3"],
                    l3_sequences,
                    l3_labels,
                    sample_indices,
                    agent_concurrency,
                )

                if baseline_seconds is None:
                    first_stats = _latency_stats(replay["events"])
                    baseline_seconds = first_stats["p50_observed_seconds"]
                    baseline_by_local_index = {
                        int(event["local_index"]): float(event["observed_latency_seconds"])
                        for event in replay["events"]
                        if int(event.get("agent_id", 0)) == 0
                    }

                latency = _latency_stats(
                    replay["events"],
                    baseline_seconds=baseline_seconds,
                    baseline_by_local_index=baseline_by_local_index,
                )

                final = np.array(final_before_l3, dtype=int)
                scores = np.array(scores_before_l3, dtype=float)
                p3 = replay["predictions"] if len(idx_to_l3) else np.array([], dtype=int)
                s3 = replay["probabilities"] if len(idx_to_l3) else np.array([], dtype=float)
                if len(idx_to_l3):
                    final[idx_to_l3[p3 == 1]] = 1
                    scores[idx_to_l3] = s3

                metrics = compute_complete_metrics(labels, final, scores)
                metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
                metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

                layer_stats = dict(gd1_artifact["layer_stats"])
                layer_stats["layer3"] = {
                    "processed": int(len(idx_to_l3)),
                    "blocked": int(p3.sum()) if len(idx_to_l3) else 0,
                    "forwarded": int((p3 == 0).sum()) if len(idx_to_l3) else 0,
                    "time": float(replay["wall_time_seconds"]),
                    "backend_concurrency": int(agent_concurrency),
                    "agent_concurrency": int(agent_concurrency),
                    "agent_streams": replay["agent_streams"],
                    "latency": latency,
                }
                pipeline._attach_layer_confusion(layer_stats, "layer3", l3_labels, p3)

                l1_l2_time = float(gd1_artifact["timing_components"]["l1_l2_wall_time_seconds"])
                l3_time = float(replay["wall_time_seconds"])
                total_time = float(l1_l2_time + l3_time)
                num_samples = int(gd1_artifact["num_samples"])
                sample_meta = gd1_artifact["sample_metadata"]

                result = {
                    "test_mode": "high_benign_backend_latency_sweep",
                    "test_mode_id": 11,
                    "strategy": "staged_high_benign_backend_replay",
                    "predictions": final,
                    "probabilities": scores,
                    "num_samples": num_samples,
                    "num_detected": int(final.sum()),
                    "time": total_time,
                    "raw_observed_wall_time": total_time,
                    "layer_stats": layer_stats,
                    "metrics": metrics,
                    "layer1_embedding_cache_loaded": gd1_artifact["layer1_embedding_cache_loaded"],
                    "layer1_embedding_batch_size": gd1_artifact["layer1_embedding_batch_size"],
                    "layer_implementations": dict(gd1_artifact["layer_implementations"]),
                    "selected_layer3_variant": gd1_artifact["selected_layer3_variant"],
                    "sweep_metadata": {
                        **gd2_config,
                        "benign_pct": int(benign_pct),
                        "ratio": float(sample_meta["actual_benign_ratio"]),
                        "actual_benign_ratio": float(sample_meta["actual_benign_ratio"]),
                        "benign_count": int(sample_meta["benign_count"]),
                        "malicious_count": int(sample_meta["malicious_count"]),
                        "attack_type_counts": sample_meta["attack_type_counts"],
                        "run": int(run_idx),
                        "backend_concurrency": int(agent_concurrency),
                        "agent_concurrency": int(agent_concurrency),
                        "l2_processed": int(gd1_artifact["routing"]["l2_processed"]),
                        "l2_ratio": float(gd1_artifact["routing"]["l2_ratio"]),
                        "l3_candidates": int(len(idx_to_l3)),
                        "l3_processed": int(len(idx_to_l3)),
                        "l3_calls_total": int(len(idx_to_l3) * agent_concurrency),
                        "l3_ratio": float(gd1_artifact["routing"]["l3_ratio"]),
                        "baseline_source_concurrency": 1,
                        "baseline_service_seconds": float(baseline_seconds or 0.0),
                        "gd1_artifact": f"gd1/{_gd1_artifact_name(benign_pct, run_idx)}",
                        "events_file": events_file,
                        "candidates_file": f"gd1/{gd1_artifact['candidates_file']}",
                        "uses_detection_pipeline": True,
                        "save_results_format": True,
                    },
                    "timing_components": {
                        "l1_l2_wall_time_seconds": l1_l2_time,
                        "layer1_time_seconds": float(gd1_artifact["timing_components"]["layer1_time_seconds"]),
                        "layer2_time_seconds": float(gd1_artifact["timing_components"]["layer2_time_seconds"]),
                        "l3_backend_wall_time_seconds": l3_time,
                        "pipeline_total_time_seconds": total_time,
                        "pipeline_latency_per_sample_seconds": float(total_time / num_samples) if num_samples else 0.0,
                    },
                    "l3_backend_latency": {
                        **latency,
                        "observed_backend_wall_time_seconds": l3_time,
                        "baseline_source_concurrency": 1,
                        "agent_concurrency": int(agent_concurrency),
                        "agent_streams": replay["agent_streams"],
                    },
                }

                save_results(result, out_path)
                _write_jsonl(events_path, replay["events"])

                summary_rows.append(
                    _summary_row_from_result(
                        result,
                        benign_pct,
                        run_idx,
                        agent_concurrency,
                        out_file,
                    )
                )
                _write_gd2_summary(summary_rows, gd2_dir, current_dir)

    _write_gd2_summary(summary_rows, gd2_dir, current_dir)

    print("\nGD_2 completed.")
    print(f"Output: {gd2_dir}")
    print(f"Combined summary: {current_dir / 'summary.csv'}")
    return current_dir


def run_high_benign_backend_latency_sweep() -> None:
    print("\nRUNNING HIGH-BENIGN BACKEND LATENCY SWEEP\n")
    print("Staged Issue 2 flow:")
    print("  GD_1 samples high-benign workloads and freezes Mode 11 L1/L2 routing.")
    print("  GD_2 replays only Layer 3 with con=1 baseline and con=n contention.\n")

    stage_choice = _ask_choice(
        "HIGH-BENIGN BACKEND LATENCY SWEEP STAGES",
        [
            ("1", "Run GD_1 only: high-benign L1/L2 sweep, no Layer 3 calls"),
            ("2", "Run GD_2 only: Layer 3 replay from saved GD_1"),
            ("3", "Run GD_1 + GD_2"),
            ("0", "Back"),
        ],
    )
    if stage_choice == "0":
        return

    cfg = Issue2HighBenignSweepConfig
    layer3_variant = _ask_layer3_variant()
    normalized_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _current_dir(normalized_variant)

    if stage_choice in {"1", "3"}:
        gd1_exists = (current_dir / "gd1" / "config.json").exists()
        overwrite_gd1 = False
        if gd1_exists:
            overwrite_gd1 = _yes_no(
                f"Existing GD_1 high-benign workload found at {current_dir / 'gd1'}. Overwrite it?",
                False,
            )
        if gd1_exists and not overwrite_gd1:
            print(f"Keeping existing GD_1 workload: {current_dir / 'gd1'}")
        else:
            total_size = _ask_positive_int(
                f"Fixed total sample size [default: {cfg.FIXED_TOTAL_SIZE}]: ",
                str(cfg.FIXED_TOTAL_SIZE),
            )
            repeats = _ask_positive_int(f"Repeats [default: {cfg.REPEATS}]: ", str(cfg.REPEATS))
            benign_start_pct = _ask_pct(
                f"Benign start percentage [default: {cfg.BENIGN_START_PCT}]: ",
                str(cfg.BENIGN_START_PCT),
            )
            benign_end_pct = _ask_pct(
                f"Benign end percentage [default: {cfg.BENIGN_END_PCT}]: ",
                str(cfg.BENIGN_END_PCT),
            )
            benign_step_pct = _ask_positive_int(
                f"Benign step percentage [default: {cfg.BENIGN_STEP_PCT}]: ",
                str(cfg.BENIGN_STEP_PCT),
            )
            random_seed = int(_ask_with_default(f"Random seed [default: {cfg.RANDOM_SEED}]: ", str(cfg.RANDOM_SEED)))

            if benign_start_pct > benign_end_pct:
                raise ValueError("Benign start percentage must be <= benign end percentage.")

            _run_gd1(
                layer3_variant=layer3_variant,
                total_size=total_size,
                repeats=repeats,
                benign_start_pct=benign_start_pct,
                benign_end_pct=benign_end_pct,
                benign_step_pct=benign_step_pct,
                random_seed=random_seed,
                overwrite=overwrite_gd1,
            )

    if stage_choice in {"2", "3"}:
        gd2_exists = (current_dir / "gd2").exists() and any((current_dir / "gd2").glob("*.json"))
        overwrite_gd2 = False
        if gd2_exists:
            overwrite_gd2 = _yes_no(
                f"Existing GD_2 high-benign results found at {current_dir / 'gd2'}. Overwrite them?",
                False,
            )
        target_agent_concurrency = _ask_positive_int(
            "Target Layer 3 agent concurrency n [default: 4]: ",
            "4",
        )
        _run_gd2(
            layer3_variant=layer3_variant,
            target_agent_concurrency=target_agent_concurrency,
            overwrite=overwrite_gd2,
        )
