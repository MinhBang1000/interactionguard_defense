"""Audit regenerated data against its pre-change ZIP without model inference.

Run after splitting/preprocessing: ``python -m data.validate_preprocessing``.
This validates structure and provenance, not semantic completeness of labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from data.data_preprocess import (
    ROOT, RETRIEVAL_HEADERS, RETRIEVAL_FOOTER, read_jsonl, sha256, write_json,
)
from data.data_split import SOURCES, SPLITS, make_splits


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args(argv)
    manifest = json.loads((args.data_dir / "splits/split_manifest.json").read_text())
    archive = Path(manifest["membership_archive"])
    by_type = {}
    for name, kind in SOURCES.items():
        source = args.data_dir / "raw" / name
        require(sha256(source) == manifest["raw_sha256"][name], f"Raw changed: {source}")
        by_type[kind] = read_jsonl(source)
        for row in by_type[kind]:
            row["attack_type"] = kind
    fresh = make_splits(by_type, manifest["seed"],
                        manifest["requested_ratios"]["benign"], manifest["requested_ratios"]["attack"])
    report = {"archive": str(archive), "raw_sha256_verified": True,
              "integrity": manifest["integrity"], "splits": {}, "review": [],
              "limitations": [
                  "Known-payload labels are operational annotations, not an independent semantic judge.",
                  "Attack-success metadata uses a tool-result proxy; prompt attacks still require a success judge.",
                  "Shared prompts and identical pre-injection prefixes are intentional; no wholly unseen-content claim.",
                  "Correlated scenarios without both linked fragments are not complete-chain evidence.",
              ]}
    with zipfile.ZipFile(archive) as handle:
        require(handle.testzip() is None, "Backup ZIP failed CRC verification")
        for name in SOURCES:
            require(hashlib.sha256(handle.read(f"data/raw/{name}")).hexdigest() == manifest["raw_sha256"][name],
                    f"Raw differs from backup: {name}")
        for split in SPLITS:
            rows = read_jsonl(args.data_dir / f"processed/{split}_processed.jsonl")
            old = [json.loads(line) for line in handle.read(f"data/processed/{split}_processed.jsonl").decode().splitlines() if line.strip()]
            require([r["id"] for r in rows] == [r["id"] for r in old], f"Prefix membership/order changed: {split}")
            legacy_wrappers = sum(any(text.startswith(header) for header in RETRIEVAL_HEADERS) or text.endswith(RETRIEVAL_FOOTER)
                                  for row in rows for text in row["channels"]["retrieval"])
            require(legacy_wrappers == 0, f"Collection wrapper remains: {split}")
            audits = read_jsonl(args.data_dir / f"processed/{split}_episode_audit.jsonl")
            stats = json.loads((args.data_dir / f"processed/{split}_report_processed.json").read_text())
            report["splits"][split] = {
                "episodes": stats["episodes"], "prefixes": len(rows),
                "old_positive": sum(r["label"] for r in old), "new_positive": sum(r["label"] for r in rows),
                "label_changes": dict(Counter(f"{a['label']}->{b['label']}" for a, b in zip(old, rows))),
                "legacy_wrapper_occurrences": legacy_wrappers,
                "fresh_seed_membership_matches_reference": [r["id"] for r in fresh[split]] == manifest["episode_ids"][split],
                "token_lengths": stats.get("token_lengths"),
                "correlated_coverage": stats["correlated_coverage"],
                "linked_correlated_episodes": stats["linked_correlated_episodes"],
                "linked_correlated_within_safe_prefix": stats["linked_correlated_within_safe_prefix"],
                "email_redirection_episodes": stats["email_redirection_episodes"],
                "prefixes_requiring_annotation_review": stats["prefixes_requiring_annotation_review"],
                "training_eligibility": stats.get("training_eligibility"),
            }
            report["review"].extend({"id": a["id"], "split": split, "issues": a["needs_review"]} for a in audits if a["needs_review"])
    sequences = {split: {r["sequence"] for r in read_jsonl(args.data_dir / f"processed/{split}_processed.jsonl")} for split in SPLITS}
    report["intentional_exact_input_overlap"] = {f"{a}_{b}": len(sequences[a] & sequences[b])
                                                 for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
    training_dir = args.data_dir / "training"
    if (training_dir / "manifest.json").exists():
        from data.prepare_training import load_prepared
        report["prepared_training"] = {
            f"{stage}/{split}": {"rows": len(load_prepared(stage, split, output_dir=training_dir)),
                                "sha256": sha256(training_dir / stage / f"{split}.jsonl")}
            for stage in ("encoder", "autoencoder") for split in ("train", "val")}
    output = args.data_dir / "processed/preprocessing_validation.json"
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Validation written to {output}")
    return report


if __name__ == "__main__":
    main()
