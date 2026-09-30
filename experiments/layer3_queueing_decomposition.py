from __future__ import annotations

import random
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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


ATTACK_TYPES = ("prompt", "rag", "tool", "correlated")


def _current_dir(layer3_variant: Optional[str]) -> Path:
    suffix = _layer3_variant_suffix(layer3_variant)
    return RESULT_DIR / "backend_queueing_decomposition" / f"mode_11{suffix}" / "current"


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


def _ask_layer3_variant() -> Optional[str]:
    print("\nLayer 3 version")
    print("This queueing decomposition sweep uses Mode 11.")
    print("Press Enter for default/current logical layer3.\n")
    print("default) current logical layer3")
    print("orig)    Original Layer3Detector")
    print("o2)      Layer3DetectorO2")
    print("custom configured key such as layer3o2")

    import yaml

    with open(CONFIG_DIR / "detection_config.yaml", "r", encoding="utf-8") as f:
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


def _parse_benign_ratio(raw: str) -> Optional[float]:
    text = str(raw or "").strip().lower()
    if text in {"", "full", "natural", "all"}:
        return None
    value = float(text)
    if not 1 <= value <= 99:
        raise ValueError("Benign ratio must be between 1 and 99, or 'full'.")
    return value / 100.0


def _sample_fixed_ratio(
    test_data: List[Dict[str, Any]],
    benign_ratio: Optional[float],
    total_size: int,
    random_seed: int,
) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    if benign_ratio is None:
        return list(test_data), list(range(len(test_data))), {
            "mode": "full",
            "requested_benign_ratio": None,
            "actual_benign_ratio": sum(1 for row in test_data if int(row["label"]) == 0) / max(1, len(test_data)),
            "benign_count": sum(1 for row in test_data if int(row["label"]) == 0),
            "malicious_count": sum(1 for row in test_data if int(row["label"]) == 1),
            "total_size": len(test_data),
            "attack_type_counts": {
                attack_type: sum(
                    1
                    for row in test_data
                    if int(row["label"]) == 1 and row.get("attack_type") == attack_type
                )
                for attack_type in ATTACK_TYPES
            },
        }

    rng = random.Random(int(random_seed))
    benign_indices = [i for i, row in enumerate(test_data) if int(row["label"]) == 0]
    malicious_indices_by_type = {
        attack_type: [
            i
            for i, row in enumerate(test_data)
            if int(row["label"]) == 1 and row.get("attack_type") == attack_type
        ]
        for attack_type in ATTACK_TYPES
    }

    n_benign = int(round(int(total_size) * float(benign_ratio)))
    n_malicious = int(total_size) - n_benign
    if n_benign > len(benign_indices):
        raise ValueError(f"Need {n_benign} benign samples, but only {len(benign_indices)} are available.")

    selected_benign = rng.sample(benign_indices, n_benign)
    selected_malicious: List[int] = []
    per_type_target = n_malicious // len(ATTACK_TYPES)
    remaining_by_type: Dict[str, List[int]] = {}

    for attack_type in ATTACK_TYPES:
        pool = malicious_indices_by_type[attack_type]
        take = min(per_type_target, len(pool))
        chosen = rng.sample(pool, take)
        selected_malicious.extend(chosen)
        chosen_set = set(chosen)
        remaining_by_type[attack_type] = [idx for idx in pool if idx not in chosen_set]

    remaining_needed = n_malicious - len(selected_malicious)
    if remaining_needed > 0:
        remaining_pool: List[int] = []
        for attack_type in ATTACK_TYPES:
            remaining_pool.extend(remaining_by_type[attack_type])
        if remaining_needed > len(remaining_pool):
            raise ValueError(
                f"Need {remaining_needed} extra malicious samples, but only {len(remaining_pool)} remain."
            )
        selected_malicious.extend(rng.sample(remaining_pool, remaining_needed))

    selected_indices = selected_benign + selected_malicious
    rng.shuffle(selected_indices)
    sampled = [test_data[i] for i in selected_indices]
    attack_type_counts = {
        attack_type: sum(
            1
            for idx in selected_malicious
            if test_data[idx].get("attack_type") == attack_type
        )
        for attack_type in ATTACK_TYPES
    }
    return sampled, selected_indices, {
        "mode": "fixed_ratio",
        "requested_benign_ratio": benign_ratio,
        "actual_benign_ratio": n_benign / max(1, int(total_size)),
        "benign_count": n_benign,
        "malicious_count": n_malicious,
        "total_size": int(total_size),
        "attack_type_counts": attack_type_counts,
    }


def _latency_stats(
    events: List[Dict[str, Any]],
    baseline_service_seconds: Optional[float] = None,
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
            "baseline_service_seconds": baseline_service_seconds,
            "p50_serving_seconds": baseline_service_seconds or 0.0,
            "p50_queueing_seconds": 0.0,
            "p95_queueing_seconds": 0.0,
            "mean_queueing_seconds": 0.0,
            "p50_paired_queueing_seconds": None,
            "p95_paired_queueing_seconds": None,
        }

    p50_observed = float(np.percentile(values, 50))
    p95_observed = float(np.percentile(values, 95))
    service = float(baseline_service_seconds) if baseline_service_seconds is not None else p50_observed
    queue_values = np.maximum(values - service, 0.0)
    stats = {
        "count": int(values.size),
        "mean_observed_seconds": float(values.mean()),
        "p50_observed_seconds": p50_observed,
        "p95_observed_seconds": p95_observed,
        "max_observed_seconds": float(values.max()),
        "baseline_service_seconds": service,
        "p50_serving_seconds": service,
        "p50_queueing_seconds": float(max(0.0, p50_observed - service)),
        "p95_queueing_seconds": float(max(0.0, p95_observed - service)),
        "mean_queueing_seconds": float(queue_values.mean()),
        "p50_paired_queueing_seconds": None,
        "p95_paired_queueing_seconds": None,
    }

    if baseline_by_local_index:
        paired = []
        for event in events:
            local_idx = int(event["local_index"])
            if local_idx in baseline_by_local_index:
                paired.append(
                    max(
                        0.0,
                        float(event["observed_latency_seconds"]) - float(baseline_by_local_index[local_idx]),
                    )
                )
        if paired:
            paired_values = np.array(paired, dtype=float)
            stats["p50_paired_queueing_seconds"] = float(np.percentile(paired_values, 50))
            stats["p95_paired_queueing_seconds"] = float(np.percentile(paired_values, 95))

    return stats


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
        layer_stats["layer2"] = {"processed": 0, "blocked": 0, "sent_to_next": 0, "time": 0.0}

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


def _call_layer3(layer3: Any, sequence: str) -> Tuple[int, float, Dict[str, Any]]:
    if hasattr(layer3, "_call_llm"):
        result = layer3._call_llm(str(sequence or "").strip())
        confidence = float(result.get("confidence", 0.0) or 0.0)
        is_attack = bool(result.get("is_malicious", False)) and confidence >= float(getattr(layer3, "threshold", 0.5))
        return int(is_attack), confidence, result

    detected, score = layer3.detect_single(str(sequence or "").strip())
    return int(bool(detected)), float(score), {"reason": "detect_single"}


def _replay_layer3(
    layer3: Any,
    sequences: List[str],
    labels: np.ndarray,
    sample_indices: np.ndarray,
    agent_concurrency: int,
) -> Dict[str, Any]:
    events: List[Dict[str, Any]] = []
    predictions_by_agent = [np.zeros(len(sequences), dtype=int) for _ in range(agent_concurrency)]
    probabilities_by_agent = [np.zeros(len(sequences), dtype=float) for _ in range(agent_concurrency)]
    agent_streams: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()

    def agent_stream(agent_id: int):
        stream_events: List[Dict[str, Any]] = []
        stream_start = time.perf_counter()
        for local_idx, sequence in enumerate(sequences):
            label = labels[local_idx] if labels is not None else None
            t_send_epoch = time.time()
            t_send_perf = time.perf_counter()
            try:
                pred, score, detail = _call_layer3(layer3, sequence)
                error = None
            except Exception as exc:
                pred = 0
                score = 0.0
                detail = {"reason": f"Layer3 replay error: {str(exc)[:120]}"}
                error = str(exc)[:240]
            t_receive_perf = time.perf_counter()
            observed = float(t_receive_perf - t_send_perf)

            stream_events.append({
                "agent_id": int(agent_id),
                "agent_concurrency": int(agent_concurrency),
                "local_index": int(local_idx),
                "sample_index": int(sample_indices[local_idx]),
                "label": int(label) if label is not None else None,
                "t_send_epoch": float(t_send_epoch),
                "t_receive_epoch": float(t_send_epoch + observed),
                "observed_latency_seconds": observed,
                "prediction": int(pred),
                "probability": float(score),
                "error": error,
                "detail": detail,
            })

        stream_values = np.array([event["observed_latency_seconds"] for event in stream_events], dtype=float)
        return int(agent_id), stream_events, {
            "agent_id": int(agent_id),
            "call_count": int(len(stream_events)),
            "wall_time_seconds": float(time.perf_counter() - stream_start),
            "mean_observed_seconds": float(stream_values.mean()) if stream_values.size else 0.0,
            "p50_observed_seconds": float(np.percentile(stream_values, 50)) if stream_values.size else 0.0,
            "p95_observed_seconds": float(np.percentile(stream_values, 95)) if stream_values.size else 0.0,
        }

    with ThreadPoolExecutor(max_workers=agent_concurrency) as executor:
        futures = [executor.submit(agent_stream, agent_id) for agent_id in range(agent_concurrency)]
        for future in as_completed(futures):
            agent_id, stream_events, stream_summary = future.result()
            agent_streams.append(stream_summary)
            for event in stream_events:
                local_idx = int(event["local_index"])
                predictions_by_agent[agent_id][local_idx] = event["prediction"]
                probabilities_by_agent[agent_id][local_idx] = event["probability"]
            events.extend(stream_events)

    events.sort(key=lambda row: (row["agent_id"], row["local_index"]))
    agent_streams.sort(key=lambda row: row["agent_id"])
    return {
        "predictions": predictions_by_agent[0] if predictions_by_agent else np.array([], dtype=int),
        "probabilities": probabilities_by_agent[0] if probabilities_by_agent else np.array([], dtype=float),
        "events": events,
        "agent_streams": agent_streams,
        "wall_time_seconds": float(time.perf_counter() - wall_start),
    }


def _gd2_result_name(agent_concurrency: int) -> str:
    return f"agentc{int(agent_concurrency)}.json"


def _gd2_events_name(agent_concurrency: int) -> str:
    return f"l3_events_agentc{int(agent_concurrency)}.jsonl"


def _load_baseline(gd2_dir: Path) -> Tuple[Optional[float], Optional[Dict[int, float]]]:
    result_path = gd2_dir / _gd2_result_name(1)
    events_path = gd2_dir / _gd2_events_name(1)
    if not (result_path.exists() and events_path.exists()):
        return None, None
    result = _read_json(result_path)
    events = load_jsonl(events_path)
    latency = result.get("l3_backend_latency", {}) or {}
    service = latency.get("p50_serving_seconds") or latency.get("p50_observed_seconds")
    if service is None:
        service = _latency_stats(events)["p50_observed_seconds"]
    baseline_by_local_index = {
        int(event["local_index"]): float(event["observed_latency_seconds"])
        for event in events
        if int(event.get("agent_id", 0)) == 0
    }
    return float(service), baseline_by_local_index


def _write_summary(rows: List[Dict[str, Any]], gd2_dir: Path, current_dir: Path) -> None:
    if not rows:
        return
    summary = pd.DataFrame(rows).sort_values("agent_concurrency")
    summary.to_csv(gd2_dir / "summary.csv", index=False)
    summary.to_csv(current_dir / "summary.csv", index=False)


def _summary_row_from_result(result: Dict[str, Any], result_file: str) -> Dict[str, Any]:
    metadata = result.get("sweep_metadata", {}) or {}
    timing = result.get("timing_components", {}) or {}
    latency = result.get("l3_backend_latency", {}) or {}
    metrics = result.get("metrics", {}) or {}
    return {
        "agent_concurrency": int(metadata.get("agent_concurrency", 0) or 0),
        "threshold": float(metadata.get("threshold", 0.0) or 0.0),
        "benign_ratio": metadata.get("requested_benign_ratio"),
        "actual_benign_ratio": float(metadata.get("actual_benign_ratio", 0.0) or 0.0),
        "layer3_variant": metadata.get("layer3_variant", "default"),
        "num_samples": int(result.get("num_samples", 0) or 0),
        "l2_processed": int(metadata.get("l2_processed", 0) or 0),
        "l2_ratio": float(metadata.get("l2_ratio", 0.0) or 0.0),
        "l3_processed": int(metadata.get("l3_processed", 0) or 0),
        "l3_calls_total": int(metadata.get("l3_calls_total", 0) or 0),
        "l3_ratio": float(metadata.get("l3_ratio", 0.0) or 0.0),
        "l1_l2_wall_time_seconds": float(timing.get("l1_l2_wall_time_seconds", 0.0) or 0.0),
        "l3_backend_wall_time_seconds": float(timing.get("l3_backend_wall_time_seconds", 0.0) or 0.0),
        "pipeline_total_time_seconds": float(timing.get("pipeline_total_time_seconds", 0.0) or 0.0),
        "pipeline_latency_per_sample_seconds": float(timing.get("pipeline_latency_per_sample_seconds", 0.0) or 0.0),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("f1_score", metrics.get("f1", 0.0)) or 0.0),
        "count": int(latency.get("count", 0) or 0),
        "mean_observed_seconds": float(latency.get("mean_observed_seconds", 0.0) or 0.0),
        "p50_observed_seconds": float(latency.get("p50_observed_seconds", 0.0) or 0.0),
        "p95_observed_seconds": float(latency.get("p95_observed_seconds", 0.0) or 0.0),
        "p50_serving_seconds": float(latency.get("p50_serving_seconds", 0.0) or 0.0),
        "p50_queueing_seconds": float(latency.get("p50_queueing_seconds", 0.0) or 0.0),
        "p95_queueing_seconds": float(latency.get("p95_queueing_seconds", 0.0) or 0.0),
        "mean_queueing_seconds": float(latency.get("mean_queueing_seconds", 0.0) or 0.0),
        "result_file": result_file,
    }


def _run_gd1(
    layer3_variant: Optional[str],
    threshold: float,
    benign_ratio: Optional[float],
    total_size: int,
    random_seed: int,
    overwrite: bool = False,
) -> Path:
    print("\n[GD_1] Freezing fixed Mode 11 workload after Layer 1 + Layer 2.")

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
    sampled_data, selected_indices, sampling = _sample_fixed_ratio(
        test_data,
        benign_ratio=benign_ratio,
        total_size=total_size,
        random_seed=random_seed,
    )
    sequences = [row["sequence"] for row in sampled_data]
    labels = np.array([row["label"] for row in sampled_data], dtype=int)

    from models.detectors.pipeline import DetectionPipeline

    pipeline = DetectionPipeline(
        config_path=CONFIG_DIR / "detection_config.yaml",
        test_mode=11,
        layer3_variant=layer3_variant,
    )
    pipeline.layers["layer1"].threshold = float(threshold)

    partial = _run_mode11_until_layer3(pipeline, sequences, labels)
    idx_to_l3 = np.array(partial["idx_to_l3"], dtype=int)
    candidate_rows = []
    for local_idx, selected_idx in enumerate(idx_to_l3):
        source = sampled_data[int(selected_idx)]
        candidate_rows.append({
            "local_index": int(local_idx),
            "selected_index": int(selected_idx),
            "sample_index": int(selected_indices[int(selected_idx)]),
            "label": int(labels[int(selected_idx)]),
            "attack_type": str(source.get("attack_type", "")),
            "sequence": source["sequence"],
        })
    _write_jsonl(gd1_dir / "l3_candidates.jsonl", candidate_rows)

    layer1_stats = partial["layer_stats"].get("layer1", {})
    l2_processed = int(layer1_stats.get("sent_to_next", 0))
    l3_processed = int(len(idx_to_l3))
    l2_ratio = l2_processed / max(1, len(sequences))
    l3_ratio = l3_processed / max(1, len(sequences))

    artifact = {
        "stage": "GD_1",
        "sweep_type": "layer3_queueing_decomposition",
        "mode_id": 11,
        "threshold": float(threshold),
        "layer3_variant": layer3_variant or "default",
        "random_seed": int(random_seed),
        "sampling": sampling,
        "selected_sample_indices": [int(i) for i in selected_indices],
        "num_samples": int(len(sequences)),
        "labels": labels.tolist(),
        "final_before_l3": np.array(partial["final_before_l3"], dtype=int).tolist(),
        "scores_before_l3": np.array(partial["scores"], dtype=float).tolist(),
        "idx_to_l3": idx_to_l3.tolist(),
        "layer_stats": partial["layer_stats"],
        "layer_implementations": dict(pipeline.layer_aliases),
        "selected_layer3_variant": pipeline.layer_aliases.get("layer3", "layer3"),
        "layer1_embedding_cache_loaded": partial["layer1_embedding_cache_loaded"],
        "layer1_embedding_batch_size": partial["layer1_embedding_batch_size"],
        "candidates_file": "l3_candidates.jsonl",
        "routing": {
            "l2_processed": l2_processed,
            "l2_ratio": float(l2_ratio),
            "l3_processed": l3_processed,
            "l3_ratio": float(l3_ratio),
            "blocked_before_l3": int(np.array(partial["final_before_l3"], dtype=int).sum()),
        },
        "timing_components": {
            "l1_l2_wall_time_seconds": float(partial["l1_l2_wall_time_seconds"]),
            "layer1_time_seconds": float(partial["layer1_time_seconds"]),
            "layer2_time_seconds": float(partial["layer2_time_seconds"]),
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_json(gd1_dir / "workload.json", artifact)
    _write_json(current_dir / "sweep_config.json", artifact)
    pd.DataFrame([{
        "threshold": float(threshold),
        "layer3_variant": layer3_variant or "default",
        "num_samples": int(len(sequences)),
        "actual_benign_ratio": float(sampling["actual_benign_ratio"]),
        "l2_processed": l2_processed,
        "l2_ratio": float(l2_ratio),
        "l3_processed": l3_processed,
        "l3_ratio": float(l3_ratio),
        "l1_l2_wall_time_seconds": float(partial["l1_l2_wall_time_seconds"]),
    }]).to_csv(gd1_dir / "summary.csv", index=False)

    print("\n✅ GD_1 completed.")
    print(f"Output: {gd1_dir}")
    print(f"L3 candidates per agent: {l3_processed}")
    return current_dir


def _run_gd2(layer3_variant: Optional[str], max_agent_concurrency: int, overwrite: bool = False) -> Path:
    print("\n[GD_2] Replaying fixed Layer 3 workload for con=1..N.")
    if overwrite:
        print("[GD_2] Existing GD_2 artifacts will be overwritten.\n")
    else:
        print("[GD_2] Completed artifacts are skipped; choose overwrite to rerun from scratch.\n")

    layer3_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _current_dir(layer3_variant)
    gd1_dir = current_dir / "gd1"
    gd2_dir = current_dir / "gd2"
    if not (gd1_dir / "workload.json").exists():
        raise FileNotFoundError(f"GD_1 workload not found: {gd1_dir / 'workload.json'}")
    if overwrite and gd2_dir.exists():
        shutil.rmtree(gd2_dir)
    gd2_dir.mkdir(parents=True, exist_ok=True)

    gd1 = _read_json(gd1_dir / "workload.json")
    candidate_rows = load_jsonl(gd1_dir / gd1["candidates_file"])
    l3_sequences = [row["sequence"] for row in candidate_rows]
    l3_labels = np.array([row["label"] for row in candidate_rows], dtype=int)
    sample_indices = np.array([row["sample_index"] for row in candidate_rows], dtype=int)
    labels = np.array(gd1["labels"], dtype=int)
    final_before_l3 = np.array(gd1["final_before_l3"], dtype=int)
    scores_before_l3 = np.array(gd1["scores_before_l3"], dtype=float)
    idx_to_l3 = np.array(gd1["idx_to_l3"], dtype=int)
    concurrency_values = list(range(1, int(max_agent_concurrency) + 1))

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

    baseline_service = None
    baseline_by_local_index = None
    summary_rows: List[Dict[str, Any]] = []

    for agent_concurrency in concurrency_values:
        result_file = _gd2_result_name(agent_concurrency)
        events_file = _gd2_events_name(agent_concurrency)
        result_path = gd2_dir / result_file
        events_path = gd2_dir / events_file
        print(f"  -> agent concurrency={agent_concurrency}, L3 candidates per agent={len(l3_sequences)}")

        if result_path.exists() and events_path.exists():
            print(f"     skip existing: {result_file}")
            if agent_concurrency == 1 and baseline_service is None:
                baseline_service, baseline_by_local_index = _load_baseline(gd2_dir)
            summary_rows.append(_summary_row_from_result(_read_json(result_path), result_file))
            _write_summary(summary_rows, gd2_dir, current_dir)
            continue

        if baseline_service is None and agent_concurrency != 1:
            baseline_service, baseline_by_local_index = _load_baseline(gd2_dir)
            if baseline_service is None:
                raise RuntimeError("Baseline con=1 is required before running higher concurrency.")

        replay = _replay_layer3(
            pipeline.layers["layer3"],
            l3_sequences,
            l3_labels,
            sample_indices,
            int(agent_concurrency),
        )
        if agent_concurrency == 1:
            first_stats = _latency_stats(replay["events"])
            baseline_service = first_stats["p50_observed_seconds"]
            baseline_by_local_index = {
                int(event["local_index"]): float(event["observed_latency_seconds"])
                for event in replay["events"]
                if int(event.get("agent_id", 0)) == 0
            }

        latency = _latency_stats(
            replay["events"],
            baseline_service_seconds=baseline_service,
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

        l1_l2_time = float(gd1["timing_components"]["l1_l2_wall_time_seconds"])
        l3_time = float(replay["wall_time_seconds"])
        total_time = float(l1_l2_time + l3_time)
        num_samples = int(gd1["num_samples"])

        layer_stats = dict(gd1["layer_stats"])
        layer_stats["layer3"] = {
            "processed": int(len(idx_to_l3)),
            "blocked": int(p3.sum()) if len(idx_to_l3) else 0,
            "forwarded": int((p3 == 0).sum()) if len(idx_to_l3) else 0,
            "time": l3_time,
            "agent_concurrency": int(agent_concurrency),
            "agent_streams": replay["agent_streams"],
            "latency": latency,
        }
        pipeline._attach_layer_confusion(layer_stats, "layer3", l3_labels, p3)

        metadata = {
            "stage": "GD_2",
            "sweep_type": "layer3_queueing_decomposition",
            "mode_id": 11,
            "threshold": float(gd1["threshold"]),
            "requested_benign_ratio": gd1["sampling"].get("requested_benign_ratio"),
            "actual_benign_ratio": float(gd1["sampling"]["actual_benign_ratio"]),
            "layer3_variant": layer3_variant or "default",
            "agent_concurrency": int(agent_concurrency),
            "l2_processed": int(gd1["routing"]["l2_processed"]),
            "l2_ratio": float(gd1["routing"]["l2_ratio"]),
            "l3_processed": int(len(idx_to_l3)),
            "l3_calls_total": int(len(idx_to_l3) * int(agent_concurrency)),
            "l3_ratio": float(gd1["routing"]["l3_ratio"]),
            "baseline_source_concurrency": 1,
            "baseline_service_seconds": float(baseline_service),
            "events_file": events_file,
            "candidates_file": f"gd1/{gd1['candidates_file']}",
        }

        result = {
            "test_mode": "layer1_to_layer2_to_layer3_queueing_decomposition",
            "test_mode_id": 11,
            "strategy": "fixed_workload_layer3_concurrency_replay",
            "predictions": final,
            "probabilities": scores,
            "num_samples": num_samples,
            "num_detected": int(final.sum()),
            "time": total_time,
            "raw_observed_wall_time": total_time,
            "layer_stats": layer_stats,
            "metrics": metrics,
            "layer_implementations": dict(gd1["layer_implementations"]),
            "selected_layer3_variant": gd1["selected_layer3_variant"],
            "sweep_metadata": metadata,
            "timing_components": {
                "l1_l2_wall_time_seconds": l1_l2_time,
                "layer1_time_seconds": float(gd1["timing_components"]["layer1_time_seconds"]),
                "layer2_time_seconds": float(gd1["timing_components"]["layer2_time_seconds"]),
                "l3_backend_wall_time_seconds": l3_time,
                "pipeline_total_time_seconds": total_time,
                "pipeline_latency_per_sample_seconds": float(total_time / num_samples) if num_samples else 0.0,
            },
            "l3_backend_latency": {
                **latency,
                "observed_backend_wall_time_seconds": l3_time,
                "agent_concurrency": int(agent_concurrency),
                "agent_streams": replay["agent_streams"],
            },
        }
        save_results(result, result_path)
        _write_jsonl(events_path, replay["events"])
        summary_rows.append(_summary_row_from_result(result, result_file))
        _write_summary(summary_rows, gd2_dir, current_dir)

    _write_summary(summary_rows, gd2_dir, current_dir)
    print("\n✅ GD_2 completed.")
    print(f"Output: {gd2_dir}")
    print(f"Combined summary: {current_dir / 'summary.csv'}")
    return current_dir


def run_layer3_queueing_decomposition_sweep() -> None:
    print("\n🚀 RUNNING LAYER 3 QUEUEING DECOMPOSITION SWEEP\n")
    print("Staged Issue 2 flow:")
    print("  GD_1 freezes one Mode 11 workload after Layer 1/2.")
    print("  GD_2 replays the same Layer 3 workload for con=1..N.")
    print("  Serving time is estimated from p50 latency at con=1.")
    print("  Queueing/contention time is max(0, p50 observed latency - p50 serving time).\n")

    stage_choice = _ask_choice(
        "LAYER 3 QUEUEING DECOMPOSITION STAGES",
        [
            ("1", "Run GD_1 only: fixed L1/L2 workload, no Layer 3 replay"),
            ("2", "Run GD_2 only: replay saved workload with con=1..N"),
            ("3", "Run GD_1 + GD_2"),
            ("0", "Back"),
        ],
    )
    if stage_choice == "0":
        return

    layer3_variant = _ask_layer3_variant()
    normalized_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _current_dir(normalized_variant)

    if stage_choice in {"1", "3"}:
        gd1_exists = (current_dir / "gd1" / "workload.json").exists()
        overwrite_gd1 = False
        if gd1_exists:
            overwrite_gd1 = _yes_no(
                f"Existing GD_1 workload found at {current_dir / 'gd1'}. Overwrite it?",
                False,
            )
        if gd1_exists and not overwrite_gd1:
            print(f"Keeping existing GD_1 workload: {current_dir / 'gd1'}")
        else:
            threshold = float(_ask_with_default("Fixed Layer 1 threshold [default: 0.10]: ", "0.10"))
            benign_ratio_raw = _ask_with_default("Fixed benign ratio percent, or 'full' [default: full]: ", "full")
            benign_ratio = _parse_benign_ratio(benign_ratio_raw)
            default_total = "2772"
            total_size = len(load_jsonl(PROCESSED_DIR / "test_processed.jsonl"))
            if benign_ratio is not None:
                total_size = _ask_positive_int(
                    f"Total samples for fixed-ratio workload [default: {default_total}]: ",
                    default_total,
                )
            random_seed = int(_ask_with_default("Random seed [default: 42]: ", "42"))
            _run_gd1(
                layer3_variant=layer3_variant,
                threshold=threshold,
                benign_ratio=benign_ratio,
                total_size=total_size,
                random_seed=random_seed,
                overwrite=overwrite_gd1,
            )

    if stage_choice in {"2", "3"}:
        gd2_exists = (current_dir / "gd2").exists() and any((current_dir / "gd2").glob("agentc*.json"))
        overwrite_gd2 = False
        if gd2_exists:
            overwrite_gd2 = _yes_no(
                f"Existing GD_2 results found at {current_dir / 'gd2'}. Overwrite them?",
                False,
            )
        max_agent_concurrency = _ask_positive_int("Max agent concurrency N [default: 4]: ", "4")
        _run_gd2(
            layer3_variant=layer3_variant,
            max_agent_concurrency=max_agent_concurrency,
            overwrite=overwrite_gd2,
        )
