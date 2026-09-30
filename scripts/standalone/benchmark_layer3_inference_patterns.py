# ============================================================================
# FILE: eval/benchmark_layer3_inference_patterns.py
# Benchmark Layer 3 inference latency and generation patterns across Ollama models.
#
# Purpose:
# - Compares a baseline Layer 3 model and a fine-tuned Layer 3 model on the
#   same processed traces to explain latency gaps and output-pattern changes.
#
# Workflow:
# - Loads processed test traces from `data/processed/test_processed.jsonl`.
# - Samples traces with optional attack-type filters.
# - Sends the same Layer 3 system prompt and user trace to multiple Ollama
#   models, measures latency, and records raw outputs and token usage.
# - Saves both per-sample comparisons and aggregate summaries to `eval/`.
#
# Use this file when:
# - You want to understand why a fine-tuned model is slower than the original.
# - You want to inspect JSON formatting, output length, confidence behavior,
#   and stop-pattern differences before optimizing deployment settings.
# ============================================================================

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI

from models.detectors.layer3_detector import Layer3Detector
from utils.paths import EVAL_DIR, PROCESSED_DIR
from utils.utils import load_jsonl


DEFAULT_MODELS = ["llama3.1:8b"]
DEFAULT_OUTPUT = EVAL_DIR / "layer3_inference_benchmark.json"


def _get_system_prompt() -> str:
    return Layer3Detector._build_system_prompt(None).strip()


def _normalize_response_content(content: str) -> str:
    text = (content or "").strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1].strip()
            if text.startswith("json"):
                text = text[4:].strip()
    return text


def _parse_json_response(content: str) -> Dict:
    normalized = _normalize_response_content(content)
    try:
        payload = json.loads(normalized)
        return {
            "valid_json": True,
            "parsed": payload,
        }
    except Exception as exc:
        return {
            "valid_json": False,
            "parsed": None,
            "error": str(exc),
        }


def _validate_layer3_schema(parsed: Optional[Dict]) -> Dict:
    if not isinstance(parsed, dict):
        return {
            "valid_schema": False,
            "schema_error": "Parsed output is not a JSON object.",
        }

    required = ["is_malicious", "confidence", "reason"]
    missing = [key for key in required if key not in parsed]
    if missing:
        return {
            "valid_schema": False,
            "schema_error": f"Missing required keys: {missing}",
        }

    if not isinstance(parsed["is_malicious"], bool):
        return {
            "valid_schema": False,
            "schema_error": "is_malicious must be boolean.",
        }
    if not isinstance(parsed["confidence"], (int, float)):
        return {
            "valid_schema": False,
            "schema_error": "confidence must be numeric.",
        }
    if not isinstance(parsed["reason"], str):
        return {
            "valid_schema": False,
            "schema_error": "reason must be a string.",
        }

    return {
        "valid_schema": True,
        "schema_error": None,
    }


def _summarize_model_results(model_name: str, rows: List[Dict]) -> Dict:
    latencies = [row["latency_seconds"] for row in rows]
    completion_tokens = [
        row["usage"]["completion_tokens"]
        for row in rows
        if row["usage"]["completion_tokens"] is not None
    ]
    output_lengths = [row["output_length_chars"] for row in rows]
    valid_json_count = sum(int(row["valid_json"]) for row in rows)
    valid_schema_count = sum(int(row["valid_schema"]) for row in rows)
    malicious_count = sum(
        int(bool(row["parsed"].get("is_malicious")))
        for row in rows
        if row["valid_schema"]
    )
    confidence_values = [
        float(row["parsed"].get("confidence"))
        for row in rows
        if row["valid_schema"] and isinstance(row["parsed"].get("confidence"), (int, float))
    ]

    per_attack_type: Dict[str, Dict] = {}
    for attack_type in sorted({row["attack_type"] for row in rows}):
        attack_rows = [row for row in rows if row["attack_type"] == attack_type]
        per_attack_type[attack_type] = {
            "count": len(attack_rows),
            "avg_latency_seconds": sum(row["latency_seconds"] for row in attack_rows) / len(attack_rows),
            "avg_output_length_chars": sum(row["output_length_chars"] for row in attack_rows) / len(attack_rows),
            "valid_json_rate": sum(int(row["valid_json"]) for row in attack_rows) / len(attack_rows),
            "valid_schema_rate": sum(int(row["valid_schema"]) for row in attack_rows) / len(attack_rows),
        }

    return {
        "model": model_name,
        "num_samples": len(rows),
        "avg_latency_seconds": sum(latencies) / len(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "max_latency_seconds": max(latencies),
        "avg_output_length_chars": sum(output_lengths) / len(output_lengths),
        "valid_json_rate": valid_json_count / len(rows),
        "valid_schema_rate": valid_schema_count / len(rows),
        "malicious_rate": malicious_count / len(rows),
        "avg_confidence": (sum(confidence_values) / len(confidence_values)) if confidence_values else None,
        "avg_completion_tokens": (sum(completion_tokens) / len(completion_tokens)) if completion_tokens else None,
        "per_attack_type": per_attack_type,
    }


def _sample_rows(
    rows: List[Dict],
    limit: int,
    attack_types: Optional[List[str]],
) -> List[Dict]:
    if attack_types:
        rows = [row for row in rows if row.get("attack_type") in attack_types]

    rows = sorted(rows, key=lambda row: row["id"])
    return rows[:limit]


def _call_model(
    client: OpenAI,
    model_name: str,
    system_prompt: str,
    sequence: str,
    num_ctx: Optional[int],
) -> Dict:
    start = time.perf_counter()
    kwargs = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sequence},
        ],
        "temperature": 0.1,
        "max_tokens": 64,
        "response_format": {"type": "json_object"},
    }
    if num_ctx is not None:
        kwargs["extra_body"] = {"options": {"num_ctx": num_ctx}}

    response = client.chat.completions.create(**kwargs)
    latency = time.perf_counter() - start
    content = response.choices[0].message.content or ""
    parsed = _parse_json_response(content)
    schema = _validate_layer3_schema(parsed.get("parsed"))
    usage = getattr(response, "usage", None)

    return {
        "latency_seconds": latency,
        "raw_output": content,
        "normalized_output": _normalize_response_content(content),
        "valid_json": parsed["valid_json"],
        "valid_schema": schema["valid_schema"],
        "schema_error": schema["schema_error"],
        "parsed": parsed.get("parsed"),
        "parse_error": parsed.get("error"),
        "usage": {
            "prompt_tokens": getattr(usage, "prompt_tokens", None) if usage is not None else None,
            "completion_tokens": getattr(usage, "completion_tokens", None) if usage is not None else None,
            "total_tokens": getattr(usage, "total_tokens", None) if usage is not None else None,
        },
        "output_length_chars": len(content),
    }


def run_benchmark(
    models: List[str],
    limit: int,
    attack_types: Optional[List[str]] = None,
    base_url: str = "http://localhost:11434/v1",
    num_ctx: Optional[int] = None,
    output_path: Path = DEFAULT_OUTPUT,
):
    traces = load_jsonl(PROCESSED_DIR / "test_processed.jsonl")
    traces = _sample_rows(traces, limit=limit, attack_types=attack_types)
    if not traces:
        raise ValueError("No traces matched the requested benchmark filters.")

    client = OpenAI(api_key="ollama", base_url=base_url)
    system_prompt = _get_system_prompt()

    per_model_rows: Dict[str, List[Dict]] = {model: [] for model in models}
    per_sample = []

    print(f"Benchmarking {len(models)} model(s) on {len(traces)} sample(s)")
    print(f"Base URL: {base_url}")
    if num_ctx is not None:
        print(f"Per-request num_ctx override: {num_ctx}")

    for index, sample in enumerate(traces, start=1):
        print(f"[{index}/{len(traces)}] {sample['id']} ({sample.get('attack_type')}, label={sample.get('label')})")
        sample_record = {
            "id": sample["id"],
            "attack_type": sample.get("attack_type"),
            "label": int(sample.get("label", 0)),
            "sequence_preview": sample["sequence"][:400],
            "models": {},
        }

        for model_name in models:
            result = _call_model(
                client=client,
                model_name=model_name,
                system_prompt=system_prompt,
                sequence=sample["sequence"],
                num_ctx=num_ctx,
            )
            row = {
                "id": sample["id"],
                "attack_type": sample.get("attack_type"),
                "label": int(sample.get("label", 0)),
                "model": model_name,
                **result,
            }
            per_model_rows[model_name].append(row)
            sample_record["models"][model_name] = {
                "latency_seconds": result["latency_seconds"],
                "valid_json": result["valid_json"],
                "valid_schema": result["valid_schema"],
                "schema_error": result["schema_error"],
                "parsed": result["parsed"],
                "usage": result["usage"],
                "output_length_chars": result["output_length_chars"],
                "raw_output": result["raw_output"],
            }
            print(
                f"  - {model_name}: {result['latency_seconds']:.2f}s | "
                f"chars={result['output_length_chars']} | "
                f"json={result['valid_json']} | "
                f"schema={result['valid_schema']} | "
                f"completion_tokens={result['usage']['completion_tokens']}"
            )

        per_sample.append(sample_record)

    summary = {
        "models": {
            model_name: _summarize_model_results(model_name, rows)
            for model_name, rows in per_model_rows.items()
        }
    }

    payload = {
        "meta": {
            "models": models,
            "num_samples": len(traces),
            "attack_types": attack_types,
            "base_url": base_url,
            "num_ctx_override": num_ctx,
            "system_prompt": system_prompt,
        },
        "summary": summary,
        "per_sample": per_sample,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    print(f"\nSaved benchmark report to: {output_path}")
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark Layer 3 inference latency and output patterns.")
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="List of Ollama model names to compare, e.g. llama3.1:8b agent-defense-v4",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of processed test samples to benchmark.",
    )
    parser.add_argument(
        "--attack-types",
        nargs="*",
        default=None,
        help="Optional attack type filters, e.g. prompt rag tool correlated.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="OpenAI-compatible Ollama endpoint.",
    )
    parser.add_argument(
        "--num-ctx",
        type=int,
        default=None,
        help="Optional per-request num_ctx override sent through Ollama options.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Path to save the JSON benchmark report.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        models=args.models,
        limit=args.limit,
        attack_types=args.attack_types,
        base_url=args.base_url,
        num_ctx=args.num_ctx,
        output_path=Path(args.output),
    )
