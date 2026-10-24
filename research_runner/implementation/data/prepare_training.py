"""Materialize fixed encoder/AE train-validation datasets during preparation.

Training loads these artifacts verbatim; filtering belongs only to this stage.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import tempfile

from research_runner.implementation.data.data_preprocess import ROOT, VERSION, read_jsonl, sha256, write_json, write_jsonl
from research_runner.implementation.data.training_policy import POLICY_VERSION, select_training_rows

OUTPUT_DIR = ROOT / "data/training"
MANIFEST_VERSION = "prepared_training_v1"
STAGES = ("encoder", "autoencoder")
SPLITS = ("train", "val")


def prepare(processed_dir=ROOT / "data/processed", output_dir=OUTPUT_DIR):
    processed_dir, output_dir = Path(processed_dir).resolve(), Path(output_dir).resolve()
    source_rows = {split: read_jsonl(processed_dir / f"{split}_processed.jsonl") for split in SPLITS}
    episode_sets = []
    for split, rows in source_rows.items():
        if not rows or len({r["id"] for r in rows}) != len(rows):
            raise ValueError(f"Empty or duplicate source rows: {split}")
        if any(type(r.get("label")) is not int or r["label"] not in (0, 1) for r in rows):
            raise ValueError(f"Invalid labels: {split}")
        episode_sets.append({r["original_trace_id"] for r in rows})
    if episode_sets[0] & episode_sets[1]:
        raise ValueError("Train/validation episode IDs overlap")
    selected = {(stage, split): select_training_rows(rows, benign_only=stage == "autoencoder")
                for stage in STAGES for split, rows in source_rows.items()}
    for (stage, split), rows in selected.items():
        if not rows:
            raise ValueError(f"No eligible rows for {stage}/{split}")
        if stage == "encoder" and {r["label"] for r in rows} != {0, 1}:
            raise ValueError(f"Encoder {split} requires both labels")
    manifest = {"version": MANIFEST_VERSION, "preprocessing_version": VERSION,
                "policy_version": POLICY_VERSION,
                "policy": {"excluded_attack_types": ["correlated"], "exclude_annotation_requires_review": True,
                           "encoder_labels": [0, 1], "autoencoder_labels": [0],
                           "sampling": "all_eligible_prefixes_in_source_order;_no_resplit_or_resample"},
                "sources": {}, "artifacts": {}}
    for split, rows in source_rows.items():
        source = processed_dir / f"{split}_processed.jsonl"
        manifest["sources"][split] = {"path": str(source), "sha256": sha256(source), "rows": len(rows),
                                       "correlated_rows": sum(r.get("attack_type") == "correlated" for r in rows),
                                       "unresolved_rows": sum(r.get("annotation_requires_review", True) for r in rows)}
    # Publish manifest last. Interrupted replacement is detected by hash checks.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="prepared_training_", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for (stage, split), rows in selected.items():
            relative = f"{stage}/{split}.jsonl"
            write_jsonl(staging / relative, rows)
            manifest["artifacts"][relative] = {"sha256": sha256(staging / relative), "rows": len(rows),
                                                "episodes": len({r["original_trace_id"] for r in rows}),
                                                "labels": dict(Counter(str(r["label"]) for r in rows))}
        for relative in manifest["artifacts"]:
            target = output_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            (staging / relative).replace(target)
        write_json(staging / "manifest.json", manifest)
        (staging / "manifest.json").replace(output_dir / "manifest.json")
    for stage in STAGES:
        for split in SPLITS:
            meta = manifest["artifacts"][f"{stage}/{split}.jsonl"]
            print(f"Prepared {stage}/{split}: {meta['rows']} prefixes; labels={meta['labels']}", flush=True)
    return manifest


def load_prepared(stage, split, *, output_dir=OUTPUT_DIR, expected_path=None):
    """Validate provenance and load verbatim. Never filter or rebuild here."""
    if stage not in STAGES or split not in SPLITS:
        raise ValueError("Expected encoder/autoencoder and train/val")
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise ValueError("Prepared training data missing. Run Data preparation 10 -> 3 (or 10 -> 5).")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("version") != MANIFEST_VERSION or manifest.get("preprocessing_version") != VERSION
            or manifest.get("policy_version") != POLICY_VERSION):
        raise ValueError("Prepared training policy/version is stale; rerun data preparation")
    relative = f"{stage}/{split}.jsonl"
    path = output_dir / relative
    if expected_path is not None and Path(expected_path).resolve() != path.resolve():
        raise ValueError("Training path must match the prepared manifest; regenerate data instead of bypassing it")
    source = manifest["sources"][split]
    if not Path(source["path"]).exists() or sha256(Path(source["path"])) != source["sha256"]:
        raise ValueError("Processed source changed since training-data preparation; rerun 10 -> 3/5")
    artifact = manifest["artifacts"][relative]
    if not path.exists() or sha256(path) != artifact["sha256"]:
        raise ValueError(f"Prepared training file missing or modified: {path}")
    rows = read_jsonl(path)
    if len(rows) != artifact["rows"]:
        raise ValueError(f"Prepared row-count mismatch: {path}")
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)
    return prepare(args.processed_dir, args.output_dir)


if __name__ == "__main__":
    main()
