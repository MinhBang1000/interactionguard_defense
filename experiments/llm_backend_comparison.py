from __future__ import annotations

import copy
import csv
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from experiments.common import (
    ask_choice as _ask_choice,
    ask_with_default as _ask_with_default,
    default_ollama_base_url as _default_ollama_base_url,
    load_detection_config as _load_detection_config,
    read_json as _read_json,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
    yes_no as _yes_no,
)
from models.detectors.layer3_detector_o2 import Layer3DetectorO2
from models.detectors.pipeline import DetectionPipeline
from utils.flow_analyzer import analyze_flow, generate_flow_diagram_data
from utils.metrics import (
    compute_complete_metrics,
    compute_precision_recall_curve_data,
    compute_roc_curve_data,
)
from utils.paths import CONFIG_DIR, PROCESSED_DIR, RESULT_DIR
from utils.utils import load_jsonl, save_results


@dataclass(frozen=True)
class BackendCandidate:
    backend: str
    model: str
    name: str
    access: str
    paper_model_id: str
    base_url: Optional[str] = None


OLLAMA_CANDIDATES = [
    BackendCandidate("ollama", "llama3.1:8b", "Llama 3.1 8B", "Ollama", "llama3.1:8b"),
    BackendCandidate("ollama", "mistral:7b", "Mistral 7B", "Ollama", "mistralai/mistral-7b"),
    BackendCandidate("ollama", "gemma2:9b", "Gemma 2 9B", "Ollama", "google/gemma-2-9b"),
]

OPENROUTER_CANDIDATES = [
    BackendCandidate("openrouter", "google/gemini-2.5-flash", "Gemini 2.5 Flash", "OpenRouter", "google/gemini-2.5-flash"),
    BackendCandidate("openrouter", "openai/gpt-5.4", "GPT-5.4", "OpenRouter", "openai/gpt-5.4"),
    BackendCandidate("openrouter", "openai/gpt-4o", "GPT-4o", "OpenRouter", "openai/gpt-4o"),
    BackendCandidate("openrouter", "openai/gpt-4.1-mini", "GPT-4.1-mini", "OpenRouter", "openai/gpt-4.1-mini"),
]

SUPPORTED_MODES = {11, 12, 13, 14}


def _parse_modes(raw: str) -> List[int]:
    modes = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        mode = int(part)
        if mode not in SUPPORTED_MODES:
            raise ValueError(f"Unsupported mode {mode}; choose from 11, 12, 13, 14.")
        modes.append(mode)
    return sorted(set(modes))


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip().lower()).strip("_")
    return slug or "unknown"


def _load_test_data(test_percent: int, random_seed: int) -> List[Dict[str, Any]]:
    test_data = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")
    if not test_data:
        raise ValueError("Empty test_processed.jsonl")

    if not 1 <= int(test_percent) <= 100:
        raise ValueError("Test percentage must be in [1, 100].")

    if int(test_percent) < 100:
        rng = random.Random(int(random_seed))
        test_data = list(test_data)
        rng.shuffle(test_data)
        subset_size = max(1, int(len(test_data) * (int(test_percent) / 100.0)))
        test_data = test_data[:subset_size]
        print(f"Using shuffled test subset: {subset_size} samples")

    return test_data


def _freeze_mode_to_l3(
    mode_id: int,
    pipeline: DetectionPipeline,
    sequences: List[str],
    labels: np.ndarray,
) -> Dict[str, Any]:
    n = len(sequences)
    scores = np.zeros(n, dtype=float)
    final_before_l3 = np.zeros(n, dtype=int)
    layer_stats: Dict[str, Dict[str, Any]] = {}
    debug: Dict[str, Any] = {}
    layer1_cache_loaded = False
    layer1_batch_size = None

    stage_start = time.perf_counter()

    if mode_id == 11:
        t1 = time.perf_counter()
        out1 = pipeline._run_layer("layer1", sequences, batch_size=128)
        t1_elapsed = time.perf_counter() - t1
        p1 = out1["predictions"]
        scores[:] = out1["probabilities"]
        idx_susp = np.where(p1 == 1)[0]
        layer1_cache_loaded = bool(out1["raw"].get("used_cached_embeddings", False))
        layer1_batch_size = int(out1["raw"].get("embedding_batch_size", 128))
        layer_stats["layer1"] = {
            "processed": n,
            "forwarded": int((p1 == 0).sum()),
            "sent_to_next": int((p1 == 1).sum()),
            "time": float(t1_elapsed),
        }
        pipeline._attach_layer_confusion(layer_stats, "layer1", labels, p1)

        idx_to_l3 = np.array([], dtype=int)
        t2_elapsed = 0.0
        if len(idx_susp) > 0:
            t2 = time.perf_counter()
            out2 = pipeline._run_layer("layer2", [sequences[i] for i in idx_susp], batch_size=4)
            t2_elapsed = time.perf_counter() - t2
            p2 = out2["predictions"]
            scores[idx_susp] = out2["probabilities"]
            final_before_l3[idx_susp[p2 == 1]] = 1
            idx_to_l3 = idx_susp[p2 == 0]
            layer_stats["layer2"] = {
                "processed": int(len(idx_susp)),
                "blocked": int((p2 == 1).sum()),
                "sent_to_next": int((p2 == 0).sum()),
                "time": float(t2_elapsed),
            }
            pipeline._attach_layer_confusion(layer_stats, "layer2", labels[idx_susp], p2)
        else:
            layer_stats["layer2"] = {"processed": 0, "blocked": 0, "sent_to_next": 0, "time": 0.0}
        debug = {"idx_suspicious_from_l1": idx_susp.tolist(), "idx_sent_to_l3_from_l2": idx_to_l3.tolist()}
        layer_times = {"layer1_time_seconds": t1_elapsed, "layer2_time_seconds": t2_elapsed}

    elif mode_id == 12:
        t2 = time.perf_counter()
        out2 = pipeline._run_layer("layer2", sequences, batch_size=4)
        t2_elapsed = time.perf_counter() - t2
        p2 = out2["predictions"]
        scores[:] = out2["probabilities"]
        idx_block = np.where(p2 == 1)[0]
        idx_to_l1 = np.where(p2 == 0)[0]
        final_before_l3[idx_block] = 1
        layer_stats["layer2"] = {
            "processed": n,
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
            "time": float(t2_elapsed),
        }
        pipeline._attach_layer_confusion(layer_stats, "layer2", labels, p2)

        idx_to_l3 = np.array([], dtype=int)
        t1_elapsed = 0.0
        if len(idx_to_l1) > 0:
            t1 = time.perf_counter()
            out1 = pipeline._run_layer("layer1", [sequences[i] for i in idx_to_l1], batch_size=128)
            t1_elapsed = time.perf_counter() - t1
            p1 = out1["predictions"]
            scores[idx_to_l1] = out1["probabilities"]
            idx_to_l3 = idx_to_l1[p1 == 1]
            layer1_cache_loaded = bool(out1["raw"].get("used_cached_embeddings", False))
            layer1_batch_size = int(out1["raw"].get("embedding_batch_size", 128))
            layer_stats["layer1"] = {
                "processed": int(len(idx_to_l1)),
                "forwarded": int((p1 == 0).sum()),
                "sent_to_next": int((p1 == 1).sum()),
                "time": float(t1_elapsed),
            }
            pipeline._attach_layer_confusion(layer_stats, "layer1", labels[idx_to_l1], p1)
        else:
            layer_stats["layer1"] = {"processed": 0, "forwarded": 0, "sent_to_next": 0, "time": 0.0}
        debug = {
            "idx_blocked_at_l2": idx_block.tolist(),
            "idx_sent_to_l1_from_l2": idx_to_l1.tolist(),
            "idx_sent_to_l3_from_l1": idx_to_l3.tolist(),
        }
        layer_times = {"layer1_time_seconds": t1_elapsed, "layer2_time_seconds": t2_elapsed}

    elif mode_id == 13:
        t1 = time.perf_counter()
        out1 = pipeline._run_layer("layer1", sequences, batch_size=128)
        t1_elapsed = time.perf_counter() - t1
        p1 = out1["predictions"]
        scores[:] = out1["probabilities"]
        idx_to_l3 = np.where(p1 == 1)[0]
        layer1_cache_loaded = bool(out1["raw"].get("used_cached_embeddings", False))
        layer1_batch_size = int(out1["raw"].get("embedding_batch_size", 128))
        layer_stats["layer1"] = {
            "processed": n,
            "forwarded": int((p1 == 0).sum()),
            "sent_to_next": int((p1 == 1).sum()),
            "time": float(t1_elapsed),
        }
        pipeline._attach_layer_confusion(layer_stats, "layer1", labels, p1)
        debug = {"idx_sent_to_l3_from_l1": idx_to_l3.tolist()}
        layer_times = {"layer1_time_seconds": t1_elapsed, "layer2_time_seconds": 0.0}

    elif mode_id == 14:
        t2 = time.perf_counter()
        out2 = pipeline._run_layer("layer2", sequences, batch_size=4)
        t2_elapsed = time.perf_counter() - t2
        p2 = out2["predictions"]
        scores[:] = out2["probabilities"]
        idx_block = np.where(p2 == 1)[0]
        idx_to_l3 = np.where(p2 == 0)[0]
        final_before_l3[idx_block] = 1
        layer_stats["layer2"] = {
            "processed": n,
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
            "time": float(t2_elapsed),
        }
        pipeline._attach_layer_confusion(layer_stats, "layer2", labels, p2)
        debug = {"idx_blocked_at_l2": idx_block.tolist(), "idx_sent_to_l3_from_l2": idx_to_l3.tolist()}
        layer_times = {"layer1_time_seconds": 0.0, "layer2_time_seconds": t2_elapsed}

    else:
        raise ValueError(f"Unsupported mode {mode_id}; choose 11, 12, 13, or 14.")

    return {
        "mode_id": int(mode_id),
        "num_samples": int(n),
        "labels": labels.tolist(),
        "final_before_l3": final_before_l3.tolist(),
        "scores_before_l3": scores.tolist(),
        "idx_to_l3": np.asarray(idx_to_l3, dtype=int).tolist(),
        "layer_stats_before_l3": layer_stats,
        "debug_before_l3": debug,
        "layer1_embedding_cache_loaded": bool(layer1_cache_loaded),
        "layer1_embedding_batch_size": layer1_batch_size,
        "layer_implementations": dict(pipeline.layer_aliases),
        "selected_layer3_variant": pipeline.layer_aliases.get("layer3", "layer3"),
        "routing_time_seconds": float(time.perf_counter() - stage_start),
        "timing_components": {
            **{key: float(value) for key, value in layer_times.items()},
            "routing_before_l3_time_seconds": float(time.perf_counter() - stage_start),
        },
    }


def _mode_dir(mode_id: int) -> Path:
    return RESULT_DIR / "llm_backend_comparison" / f"mode_{int(mode_id)}_layer3o2"


def _cache_dir(mode_id: int, test_percent: int, random_seed: int) -> Path:
    return _mode_dir(mode_id) / "routing_cache" / f"testpct_{int(test_percent):03d}_seed_{int(random_seed)}"


def _ensure_routing_cache(
    mode_id: int,
    test_percent: int,
    random_seed: int,
    force_rebuild: bool,
) -> Tuple[Path, Dict[str, Any], List[Dict[str, Any]]]:
    cache_dir = _cache_dir(mode_id, test_percent, random_seed)
    routing_path = cache_dir / "routing.json"
    candidates_path = cache_dir / "l3_candidates.jsonl"

    if routing_path.exists() and candidates_path.exists() and not force_rebuild:
        print(f"  Reusing routing cache: {cache_dir}")
        return cache_dir, _read_json(routing_path), load_jsonl(candidates_path)

    print(f"  Building routing cache for mode {mode_id}: {cache_dir}")
    test_data = _load_test_data(test_percent, random_seed)
    sequences = [item["sequence"] for item in test_data]
    labels = np.asarray([item["label"] for item in test_data], dtype=int)

    pipeline = DetectionPipeline(
        config_path=CONFIG_DIR / "detection_config.yaml",
        test_mode=mode_id,
        layer3_variant="layer3o2",
    )
    routing = _freeze_mode_to_l3(mode_id, pipeline, sequences, labels)
    idx_to_l3 = np.asarray(routing["idx_to_l3"], dtype=int)

    candidate_rows = []
    for local_idx, sample_idx in enumerate(idx_to_l3):
        source = test_data[int(sample_idx)]
        candidate_rows.append({
            "local_index": int(local_idx),
            "sample_index": int(sample_idx),
            "label": int(labels[int(sample_idx)]),
            "attack_type": str(source.get("attack_type", "")),
            "sequence": source["sequence"],
        })

    metadata = {
        "experiment": "llm_backend_comparison_routing_cache",
        "mode_id": int(mode_id),
        "layer3_variant": "layer3o2",
        "test_percent": int(test_percent),
        "random_seed": int(random_seed),
        "num_samples": int(len(sequences)),
        "l3_candidates": int(len(candidate_rows)),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "This cache freezes all non-L3 routing so candidate LLM backends can be compared without rerunning Layer 1/2.",
    }
    routing["cache_metadata"] = metadata
    routing["mode_name"] = pipeline.current_mode.get("name", "")
    _write_json(routing_path, routing)
    _write_jsonl(candidates_path, candidate_rows)
    _write_json(cache_dir / "config.json", metadata)
    return cache_dir, routing, candidate_rows


def _layer3_params_for_candidate(candidate: BackendCandidate, detection_config: Dict[str, Any]) -> Dict[str, Any]:
    layer_cfg = (
        detection_config.get("layers", {})
        .get("layer3o2", {})
        .get("params", {})
        .copy()
    )
    layer_cfg["model"] = candidate.model
    layer_cfg["backend"] = candidate.backend
    if candidate.backend == "openrouter":
        layer_cfg["base_url"] = candidate.base_url or "https://openrouter.ai/api/v1"
    else:
        layer_cfg["base_url"] = candidate.base_url or _default_ollama_base_url(detection_config)
    return layer_cfg


def _run_candidate(
    mode_id: int,
    candidate: BackendCandidate,
    routing: Dict[str, Any],
    candidate_rows: List[Dict[str, Any]],
    detection_config: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    labels = np.asarray(routing["labels"], dtype=int)
    final = np.asarray(routing["final_before_l3"], dtype=int)
    scores = np.asarray(routing["scores_before_l3"], dtype=float)
    idx_to_l3 = np.asarray(routing["idx_to_l3"], dtype=int)
    l3_sequences = [row["sequence"] for row in candidate_rows]
    l3_labels = np.asarray([row["label"] for row in candidate_rows], dtype=int)

    params = _layer3_params_for_candidate(candidate, detection_config)
    detector = Layer3DetectorO2(**params)

    t0 = time.perf_counter()
    if l3_sequences:
        out3 = detector.predict(l3_sequences)
        p3 = np.asarray(out3["predictions"], dtype=int)
        s3 = np.asarray(out3["probabilities"], dtype=float)
    else:
        out3 = {
            "predictions": np.array([], dtype=int),
            "probabilities": np.array([], dtype=float),
            "reasonings": [],
            "evidence_details": [],
        }
        p3 = np.array([], dtype=int)
        s3 = np.array([], dtype=float)
    layer3_elapsed = time.perf_counter() - t0

    if len(idx_to_l3):
        final[idx_to_l3[p3 == 1]] = 1
        scores[idx_to_l3] = s3

    layer_stats = copy.deepcopy(routing["layer_stats_before_l3"])
    layer_stats["layer3"] = {
        "processed": int(len(idx_to_l3)),
        "blocked": int((p3 == 1).sum()) if len(idx_to_l3) else 0,
        "forwarded": int((p3 == 0).sum()) if len(idx_to_l3) else 0,
        "time": float(layer3_elapsed),
        "backend": candidate.backend,
        "model": candidate.model,
        "paper_model_id": candidate.paper_model_id,
    }
    _l3_confusion = DetectionPipeline._layer_confusion(l3_labels, p3)
    if _l3_confusion is not None:
        layer_stats["layer3"]["confusion_matrix"] = _l3_confusion

    metrics = compute_complete_metrics(labels, final, scores)
    metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
    metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

    routing_time = float(routing.get("routing_time_seconds", 0.0))
    total_time = routing_time + float(layer3_elapsed)
    n = int(routing["num_samples"])

    result = {
        "test_mode": routing.get("mode_name", f"mode_{mode_id}"),
        "test_mode_id": int(mode_id),
        "strategy": "chained_llm_backend_comparison",
        "predictions": final,
        "probabilities": scores,
        "num_samples": n,
        "num_detected": int(final.sum()),
        "time": total_time,
        "raw_observed_wall_time": total_time,
        "layer_stats": layer_stats,
        "flow_analysis": analyze_flow(layer_stats, n),
        "flow_diagram": generate_flow_diagram_data(layer_stats, n),
        "metrics": metrics,
        "layer1_embedding_cache_loaded": bool(routing.get("layer1_embedding_cache_loaded", False)),
        "layer1_embedding_batch_size": routing.get("layer1_embedding_batch_size"),
        "layer_implementations": dict(routing.get("layer_implementations", {})),
        "selected_layer3_variant": "layer3o2",
        "debug": {
            **dict(routing.get("debug_before_l3", {})),
            "idx_sent_to_l3": idx_to_l3.tolist(),
        },
        "timing_components": {
            **dict(routing.get("timing_components", {})),
            "layer3_time_seconds": float(layer3_elapsed),
            "pipeline_total_time_seconds": total_time,
            "pipeline_latency_per_sample_seconds": float(total_time / n) if n else 0.0,
        },
        "llm_backend_test": {
            "backend": candidate.backend,
            "model": candidate.model,
            "name": candidate.name,
            "access": candidate.access,
            "paper_model_id": candidate.paper_model_id,
            "base_url": params.get("base_url"),
            "layer3_variant": "layer3o2",
            "routing_cache": routing.get("cache_metadata", {}),
            "result_file": out_path.name,
        },
        "layer3_raw": {
            "reasonings": out3.get("reasonings", []),
            "evidence_details": out3.get("evidence_details", []),
        },
    }
    return result


def _write_summary(mode_dir: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path = mode_dir / "summary.csv"
    fieldnames = [
        "mode_id",
        "backend",
        "model",
        "paper_model_id",
        "name",
        "num_samples",
        "l3_processed",
        "l3_ratio",
        "time",
        "latency_per_sample",
        "precision",
        "recall",
        "f1",
        "fpr",
        "fnr",
        "accuracy",
        "result_file",
        "status",
        "error",
    ]
    mode_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _result_to_summary_row(
    mode_id: int,
    candidate: BackendCandidate,
    result: Dict[str, Any],
    result_file: str,
    status: str = "ok",
    error: str = "",
) -> Dict[str, Any]:
    metrics = result.get("metrics", {}) or {}
    n = int(result.get("num_samples", 0) or 0)
    l3_processed = int(result.get("layer_stats", {}).get("layer3", {}).get("processed", 0) or 0)
    total_time = float(result.get("time", 0.0) or 0.0)
    return {
        "mode_id": int(mode_id),
        "backend": candidate.backend,
        "model": candidate.model,
        "paper_model_id": candidate.paper_model_id,
        "name": candidate.name,
        "num_samples": n,
        "l3_processed": l3_processed,
        "l3_ratio": float(l3_processed / n) if n else 0.0,
        "time": total_time,
        "latency_per_sample": float(total_time / n) if n else 0.0,
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("f1_score", metrics.get("f1", 0.0)) or 0.0),
        "fpr": float(metrics.get("fpr", 0.0) or 0.0),
        "fnr": float(metrics.get("fnr", 0.0) or 0.0),
        "accuracy": float(metrics.get("accuracy", 0.0) or 0.0),
        "result_file": result_file,
        "status": status,
        "error": error,
    }


def _select_candidates(detection_config: Dict[str, Any]) -> List[BackendCandidate]:
    ollama_base_url = _default_ollama_base_url(detection_config)
    ollama_candidates = [
        BackendCandidate(
            backend=item.backend,
            model=item.model,
            name=item.name,
            access=item.access,
            paper_model_id=item.paper_model_id,
            base_url=ollama_base_url,
        )
        for item in OLLAMA_CANDIDATES
    ]

    choice = _ask_choice(
        "LAYER 3 LLM BACKEND MODELS",
        [
            ("1", "Ollama self-hosted candidates only"),
            ("2", "OpenRouter cloud candidates only"),
            ("3", "All candidates"),
            ("4", "Custom single model"),
            ("0", "Back"),
        ],
    )
    if choice == "0":
        return []
    if choice == "1":
        return ollama_candidates
    if choice == "2":
        return OPENROUTER_CANDIDATES
    if choice == "3":
        return ollama_candidates + OPENROUTER_CANDIDATES

    backend = _ask_choice(
        "CUSTOM MODEL BACKEND",
        [("1", "Ollama"), ("2", "OpenRouter")],
    )
    backend_name = "ollama" if backend == "1" else "openrouter"
    model = input("Model id: ").strip()
    if not model:
        raise ValueError("Custom model id cannot be empty.")
    default_url = ollama_base_url if backend_name == "ollama" else "https://openrouter.ai/api/v1"
    base_url = _ask_with_default(f"Base URL [default: {default_url}]: ", default_url)
    return [
        BackendCandidate(
            backend=backend_name,
            model=model,
            name=model,
            access="Ollama" if backend_name == "ollama" else "OpenRouter",
            paper_model_id=model,
            base_url=base_url,
        )
    ]


def run_llm_backend_comparison() -> None:
    print("\nRUNNING LAYER 3 LLM BACKEND COMPARISON\n")
    print("This experiment freezes non-L3 routing once per mode, then replays Layer 3 O2")
    print("with multiple Ollama/OpenRouter models. Existing routing caches can be reused.\n")

    mode_raw = _ask_with_default("Modes to run, comma-separated [default: 11]: ", "11")
    modes = _parse_modes(mode_raw)
    test_percent = int(_ask_with_default("Test percentage [default: 100]: ", "100"))
    random_seed = int(_ask_with_default("Random seed for test subset [default: 42]: ", "42"))
    force_rebuild_cache = _yes_no("Rebuild routing cache even if present?", False)
    skip_existing = _yes_no("Skip completed model result files?", True)

    detection_config = _load_detection_config()
    candidates = _select_candidates(detection_config)
    if not candidates:
        return

    for mode_id in modes:
        mode_dir = _mode_dir(mode_id)
        mode_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "=" * 72)
        print(f"MODE {mode_id}: Layer 3 backend comparison")
        print("=" * 72)

        cache_dir, routing, candidate_rows = _ensure_routing_cache(
            mode_id=mode_id,
            test_percent=test_percent,
            random_seed=random_seed,
            force_rebuild=force_rebuild_cache,
        )
        print(f"  L3 candidates cached: {len(candidate_rows)}")

        summary_rows: List[Dict[str, Any]] = []

        for candidate in candidates:
            result_file = (
                f"mode_{mode_id}_layer3o2__{candidate.backend}__"
                f"{_slug(candidate.model)}.json"
            )
            out_path = mode_dir / result_file

            print(
                f"\n[Mode {mode_id}] {candidate.access}: "
                f"{candidate.paper_model_id} -> runtime model '{candidate.model}'"
            )

            if out_path.exists() and skip_existing:
                print(f"  skip existing: {out_path.name}")
                try:
                    existing = _read_json(out_path)
                    summary_rows.append(
                        _result_to_summary_row(mode_id, candidate, existing, result_file)
                    )
                except Exception as exc:
                    summary_rows.append(
                        _result_to_summary_row(
                            mode_id,
                            candidate,
                            {},
                            result_file,
                            status="error",
                            error=f"failed to read existing result: {str(exc)[:160]}",
                        )
                    )
                _write_summary(mode_dir, summary_rows)
                continue

            try:
                result = _run_candidate(
                    mode_id=mode_id,
                    candidate=candidate,
                    routing=routing,
                    candidate_rows=candidate_rows,
                    detection_config=detection_config,
                    out_path=out_path,
                )
                save_results(result, out_path)
                summary_rows.append(_result_to_summary_row(mode_id, candidate, result, result_file))
            except Exception as exc:
                error_payload = {
                    "test_mode_id": int(mode_id),
                    "strategy": "chained_llm_backend_comparison",
                    "error": str(exc),
                    "llm_backend_test": {
                        "backend": candidate.backend,
                        "model": candidate.model,
                        "name": candidate.name,
                        "access": candidate.access,
                        "paper_model_id": candidate.paper_model_id,
                        "routing_cache_dir": str(cache_dir),
                    },
                }
                _write_json(out_path, error_payload)
                print(f"  ERROR saved: {out_path}")
                print(f"  {str(exc)[:240]}")
                summary_rows.append(
                    _result_to_summary_row(
                        mode_id,
                        candidate,
                        error_payload,
                        result_file,
                        status="error",
                        error=str(exc)[:240],
                    )
                )

            _write_summary(mode_dir, summary_rows)

        _write_summary(mode_dir, summary_rows)
        print(f"\nSummary: {mode_dir / 'summary.csv'}")
