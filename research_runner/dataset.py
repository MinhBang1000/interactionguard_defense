"""Isolated, offline full dataset preparation using the existing data policies.

``prepare_dataset(config, output)`` writes annotated/, splits/, processed/,
training/, judges/, asr/ and dataset_manifest.json directly inside a fresh staging
directory. No raw data is copied or published. Configured paths must be absolute:
data.root is the directory containing raw/, data.judges/reviews are optional
cache inputs, and data.reference_test_sha256 optionally pins the processed test
JSONL. Existing split membership is required; it is never regenerated.

Judging defaults to skipped. Only an explicit data.judge.skip=False, pending
episodes, runtime.audit_python (or runtime.python), and paths.workspace permit a
judge subprocess. This module never imports model or training implementations.
Pending vLLM judging requires a GPU; runtime.device and max_gpu_memory_mib are
forwarded as IG_JUDGE_GPU and IG_JUDGE_MAX_MEMORY_MIB (defaults 0 and 1024).
The optional config['_dataset_final'] rebases artifact provenance for publication;
it is never a write destination. On failure the caller owns partial staging data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from research_runner.implementation.data.annotate_raw import annotate_sources
from research_runner.implementation.data.data_preprocess import (
    main as preprocess, normalize_trace, read_jsonl, sha256, write_json, write_jsonl,
)
from research_runner.implementation.data.data_split import SOURCES, SPLITS, load_membership, main as split
from research_runner.implementation.data.prepare_training import prepare
from research_runner.implementation.evaluation.asr import episode_hash, evaluate_files, load_judges
from research_runner.implementation.evaluation.prompt_success_judge import MODEL, apply_reviews, reusable_record

from .progress import emit, iterable, logged_process


def _absolute(value, name, *, preserve_symlinks=False):
    path = Path(value)
    if not path.is_absolute():
        raise ValueError(f"{name} must be an absolute, resolved path")
    return path if preserve_symlinks else path.resolve()


def _input_path(data, key):
    value = data.get(key)
    return _absolute(value, f"data.{key}") if value is not None else None


def _digest(path):
    return sha256(path) if path is not None and path.exists() else None


def _copy_cache(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source is not None and source.exists():
        shutil.copy2(source, destination)
    else:
        write_jsonl(destination, [])


def _apply_reviews(samples, judge_path, review_path):
    cached = load_judges(judge_path)
    before = json.dumps(cached, sort_keys=True)
    apply_reviews(samples, cached, review_path)
    if json.dumps(cached, sort_keys=True) != before:
        write_jsonl(judge_path, cached.values())
    return cached


def _judge(config, samples, judge_path, review_path, temporary):
    emit("dataset.judging", 0, len(samples), unit="episodes", force=True,
         message="Checking cached judgements and reviews")
    options = config["data"].get("judge", {})
    skip = options.get("skip", True)
    if type(skip) is not bool:
        raise ValueError("data.judge.skip must be a boolean")
    backend = options.get("backend", "vllm")
    if backend not in {"vllm", "openai-compatible"}:
        raise ValueError("Unsupported data.judge.backend")
    model = "meta-llama/Llama-3.1-8B-Instruct" if backend == "vllm" else MODEL
    cached = _apply_reviews(samples, judge_path, review_path)
    pending = []
    for trace in iterable(samples, "dataset.judging.cache", unit="episodes"):
        record = cached.get(trace["id"], {})
        reviewed = (record.get("episode_sha256") == episode_hash(trace)
                    and isinstance(record.get("review"), dict)
                    and record["review"].get("episode_sha256") == episode_hash(trace))
        if not reviewed and not reusable_record(trace, record, model):
            pending.append(trace)
    executed = False
    emit("dataset.judging", len(samples) - len(pending), len(samples), unit="episodes",
         message=f"{len(pending)} pending; judging {'disabled' if skip else 'enabled'}")
    if pending and not skip:
        runtime = config.get("runtime", {})
        environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
        if backend == "vllm":
            device = str(runtime.get("device", "0"))
            if device == "cpu":
                raise ValueError("Pending vLLM judging requires a GPU; runtime.device is cpu")
            if device == "cuda":
                device = "0"
            elif device.startswith("cuda:"):
                device = device.split(":", 1)[1]
            if not device.isdecimal():
                raise ValueError("vLLM runtime.device must select a GPU index")
            memory = runtime.get("max_gpu_memory_mib")
            if memory is None:
                memory = 1024
            if type(memory) is not int or memory < 1:
                raise ValueError("runtime.max_gpu_memory_mib must be a positive integer")
            environment.update(IG_JUDGE_GPU=device, IG_JUDGE_MAX_MEMORY_MIB=str(memory))
        interpreter = runtime.get("audit_python") or runtime.get("python")
        if not interpreter:
            raise ValueError("Judging requires explicit runtime.audit_python or runtime.python")
        # Resolving venv/bin/python to its target would select the wrong environment.
        interpreter = _absolute(interpreter, "runtime judge interpreter", preserve_symlinks=True)
        workspace = config.get("paths", {}).get("workspace")
        if not workspace:
            raise ValueError("Judging requires explicit paths.workspace")
        workspace = _absolute(workspace, "paths.workspace")
        if not interpreter.is_file() or not workspace.is_dir():
            raise ValueError("Judge interpreter or workspace does not exist")
        # Only unresolved episodes reach the judge; reviewed records remain intact.
        traces = temporary / "pending/prompt_injection_traces.jsonl"
        reviews = temporary / "pending/reviews.jsonl"
        write_jsonl(traces, pending)
        write_jsonl(reviews, [])
        command = [str(interpreter), "-m", "research_runner.implementation.evaluation.prompt_success_judge",
                   "--traces", str(traces), "--output", str(judge_path),
                   "--reviews", str(reviews), "--backend", backend,
                   "--base-url", options.get("base_url") or "http://127.0.0.1:11434/v1"]
        emit("dataset.judging.pending", 0, len(pending), unit="episodes", force=True,
             message="Running judge subprocess")
        logged_process(command, temporary.parent / 'judges/judge.log', phase='dataset.judge_worker',
                       cwd=workspace, check=True, env=environment)
        _apply_reviews(samples, judge_path, review_path)
        executed = True
        emit("dataset.judging.pending", len(pending), len(pending), unit="episodes",
             status="completed", force=True)
    resolved = len(samples) if executed else len(samples) - len(pending)
    emit("dataset.judging", resolved, len(samples), unit="episodes", force=True,
         status="skipped" if pending and skip else "completed",
         message=f"{len(pending) if skip else 0} episodes left pending")
    return {"skip": skip, "backend": backend, "pending_episodes": len(pending),
            "subprocess_ran": executed}


def prepare_dataset(config: dict, output: Path) -> dict:
    """Prepare a staged dataset and return its JSON-serializable hash/count manifest.

    Inputs and their split order stay immutable. Output must be absent or empty
    and disjoint from data.root and configured cache/review inputs. The returned
    manifest is also saved as output/dataset_manifest.json; root manifest.json is
    reserved for the parent's artifact store. Its artifact paths are relative to
    output (or _dataset_final once the caller publishes the directory). Any later
    edits or JSON reserialization require refreshing the affected artifact hashes.
    """
    data = config["data"]
    root = _absolute(data["root"], "data.root")
    output = Path(output).resolve()
    final = _absolute(config.get("_dataset_final", output), "_dataset_final")
    for destination in (output, final):
        if destination == root or root in destination.parents or destination in root.parents:
            raise ValueError("Dataset output/final directory must not overlap data.root")
    judges, reviews = _input_path(data, "judges"), _input_path(data, "reviews")
    for source in (judges, reviews):
        if source is not None and (source == output or output in source.parents):
            raise ValueError("Dataset output must not contain judge/review inputs")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Dataset staging output must be absent or empty")
    reference = data.get("reference_test_sha256")
    if reference is not None:
        if (not isinstance(reference, str) or len(reference) != 64
                or any(c not in "0123456789abcdefABCDEF" for c in reference)):
            raise ValueError("data.reference_test_sha256 must be a SHA-256 digest or None")
        reference = reference.lower()

    raw_hashes = {name: sha256(root / "raw" / name)
                  for name in iterable(SOURCES, "dataset.verify_raw", unit="files")}
    membership = load_membership(root, None)
    if membership is None:
        raise ValueError("Existing complete split membership is required; refusing to resplit")
    episode_count = sum(len(ids) for ids in membership.values())
    original_manifest = root / "splits/split_manifest.json"
    protected = {root / "raw" / name for name in SOURCES}
    protected.update(p for p in (judges, reviews) if p is not None)
    if original_manifest.exists():
        protected.add(original_manifest)
    else:
        protected.update(p for name in SPLITS for p in (
            root / f"splits/{name}.jsonl", root / f"processed/{name}_processed.jsonl")
                         if p.exists())
    before = {path: _digest(path) for path in protected}

    output.mkdir(parents=True, exist_ok=True)
    (output / "splits").mkdir()
    if original_manifest.exists():
        shutil.copy2(original_manifest, output / "splits/split_manifest.json")
    else:
        write_json(output / "splits/split_manifest.json", {"episode_ids": membership})
    judge_path = output / "judges/prompt_success_judgements.jsonl"
    review_path = output / "judges/prompt_success_reviews.jsonl"
    _copy_cache(judges, judge_path)
    _copy_cache(reviews, review_path)

    with tempfile.TemporaryDirectory(prefix=".dataset_inputs_", dir=output) as folder:
        temporary = Path(folder)
        (temporary / "raw").symlink_to(root / "raw", target_is_directory=True)
        (temporary / "splits").symlink_to(output / "splits", target_is_directory=True)
        samples = [normalize_trace(t) for t in read_jsonl(
            temporary / "raw/prompt_injection_traces.jsonl")]
        judge = _judge(config, samples, judge_path, review_path, temporary)
        emit("dataset.annotate", 0, episode_count, unit="episodes", force=True)
        annotation = annotate_sources(temporary, output / "annotated", judge_path)
        emit("dataset.annotate", annotation["unique_episodes"], episode_count,
             unit="episodes", status="completed", force=True)
        emit("dataset.split", 0, episode_count, unit="episodes", force=True,
             message="Reusing existing split membership and order")
        split_manifest = split(["--data-dir", str(temporary), "--source-dir",
                                str(output / "annotated"), "--require-outcomes"])
    if split_manifest["episode_ids"] != membership:
        raise ValueError("Preserved split membership/order changed")
    split_manifest["source_dir"] = str(final / "annotated")
    write_json(output / "splits/split_manifest.json", split_manifest)
    emit("dataset.split", episode_count, episode_count, unit="episodes",
         status="completed", force=True,
         message=", ".join(f"{name}={len(ids)}" for name, ids in membership.items()))
    emit("dataset.prefix", 0, unit="prefixes", force=True)
    preprocessing = preprocess(["--split-dir", str(output / "splits"),
                                "--output-dir", str(output / "processed")])
    prefix_count = sum(stats["total_samples"] for stats in preprocessing["splits"].values())
    emit("dataset.prefix", prefix_count, prefix_count, unit="prefixes",
         status="completed", force=True)
    test_hash = sha256(output / "processed/test_processed.jsonl")
    if reference is not None and test_hash != reference:
        raise ValueError(f"Processed test reference SHA-256 mismatch: {test_hash} != {reference}")
    emit("dataset.training", 0, unit="prefixes", force=True,
         message="Materializing eligible encoder/autoencoder train/validation data")
    training = prepare(output / "processed", output / "training")
    for name, source in training["sources"].items():
        source["path"] = str(final / f"processed/{name}_processed.jsonl")
    write_json(output / "training/manifest.json", training)
    training_count = sum(meta["rows"] for meta in training["artifacts"].values())
    emit("dataset.training", training_count, training_count, unit="prefixes", force=True,
         status="completed",
         message=", ".join(f"{name}={meta['rows']}" for name, meta in training["artifacts"].items()))
    emit("dataset.validate", 0, unit="files", force=True,
         message="Evaluating baseline and verifying immutable inputs and output hashes")
    report = evaluate_files(output / "splits/test.jsonl", judges=judge_path)
    report["provenance"]["traces_path"] = str(final / "splits/test.jsonl")
    write_json(output / "asr/test_baseline_asr_v3.json", report)

    if any(_digest(path) != digest for path, digest in before.items()):
        raise ValueError("Dataset preparation input changed during preparation")
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink must not be published: {path}")
        if path.is_file():
            meta = {"sha256": sha256(path), "bytes": path.stat().st_size}
            if path.suffix == ".jsonl":
                with path.open(encoding="utf-8") as handle:
                    meta["rows"] = sum(bool(line.strip()) for line in handle)
            artifacts[path.relative_to(output).as_posix()] = meta
            emit("dataset.validate", len(artifacts), unit="files",
                 message=path.relative_to(output).as_posix())
    manifest = {
        "version": "research_dataset_v1", "data_root": str(root), "output": str(final),
        "raw_sha256": raw_hashes, "membership_policy": split_manifest["membership_policy"],
        "inputs": {key: {"path": str(path) if path is not None else None,
                         "sha256": before.get(path)}
                   for key, path in (("judges", judges), ("reviews", reviews))},
        "reference_test_sha256": reference, "test_processed_sha256": test_hash,
        "counts": {
            "raw_episodes": annotation["unique_episodes"],
            "splits": {name: {"episodes": stats["episodes"], "prefixes": stats["total_samples"]}
                       for name, stats in preprocessing["splits"].items()},
            "training": {name: meta["rows"] for name, meta in training["artifacts"].items()},
        },
        "judge": judge, "artifacts": artifacts,
        "asr": {key: report[key] for key in ("contract", "global", "per_attack_type", "benign")},
    }
    write_json(output / "dataset_manifest.json", manifest)
    emit("dataset.validate", len(artifacts), len(artifacts), unit="files",
         status="completed", force=True, message="Dataset manifest written")
    return manifest
