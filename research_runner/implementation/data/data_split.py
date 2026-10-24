"""Split episodes reproducibly and normalize collection wrappers for defense.

Run ``python -m research_runner.implementation.data.data_split``. Reuse existing membership when available, or
recover it from a verified data ZIP using ``--membership-from``. Raw is immutable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from research_runner.implementation.data.data_preprocess import ROOT, VERSION, normalize_trace, read_jsonl, sha256, write_json, write_jsonl

SOURCES = {"benign_traces.jsonl": None, "prompt_injection_traces.jsonl": "prompt",
           "rag_poison_traces.jsonl": "rag", "tool_injection_traces.jsonl": "tool", "correlated_injection_traces.jsonl": "correlated"}
SPLITS = ("train", "val", "test")


def prompt_key(trace: dict) -> str:
    prompt = str(trace.get("prompt") or "").lower().strip()
    return hashlib.sha256(prompt.encode()).hexdigest() if prompt else f"empty:{trace['id']}"


def partition(rows: list, ratios: list[float], seed: int) -> tuple[list, list, list]:
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    train_end = int(len(rows) * ratios[0])
    val_end = train_end + int(len(rows) * ratios[1])
    return rows[:train_end], rows[train_end:val_end], rows[val_end:]


def load_membership(data_dir: Path, archive: Path | None) -> dict[str, list[str]] | None:
    if archive:
        result = {}
        with zipfile.ZipFile(archive) as handle:
            for split in SPLITS:
                names = [f"data/splits/{split}.jsonl", f"data/processed/{split}_processed.jsonl"]
                name = next((n for n in names if n in handle.namelist()), None)
                if name is None:
                    raise ValueError(f"No {split} membership in {archive}")
                rows = [json.loads(s) for s in handle.read(name).decode("utf-8").splitlines() if s.strip()]
                result[split] = list(dict.fromkeys(r.get("original_trace_id", r["id"]) for r in rows))
        return result
    manifest = data_dir / "splits/split_manifest.json"
    if manifest.exists():
        return json.loads(manifest.read_text())["episode_ids"]
    result = {}
    for split in SPLITS:
        candidates = [data_dir / f"splits/{split}.jsonl", data_dir / f"processed/{split}_processed.jsonl"]
        source = next((p for p in candidates if p.exists()), None)
        if source is None:
            return None
        result[split] = list(dict.fromkeys(r.get("original_trace_id", r["id"]) for r in read_jsonl(source)))
    return result


def make_splits(by_type: dict, seed: int, benign_ratios: list, attack_ratios: list) -> dict:
    benign = partition(by_type[None], benign_ratios, seed)
    prompt = partition(by_type["prompt"], attack_ratios, seed)
    tools = partition(by_type["tool"], attack_ratios, seed)
    groups = defaultdict(list)
    for row in by_type["rag"] + by_type["correlated"]:
        groups[prompt_key(row)].append(row)
    grouped = partition(list(groups.values()), attack_ratios, seed)
    result = {}
    rng = random.Random(seed)
    for i, name in enumerate(SPLITS):
        result[name] = prompt[i] + tools[i] + [r for group in grouped[i] for r in group] + benign[i]
        rng.shuffle(result[name])
    return result


def validate_splits(splits: dict, all_ids: set[str]) -> dict:
    memberships = [{r["id"] for r in splits[s]} for s in SPLITS]
    if any(len(ids) != len(splits[s]) for s, ids in zip(SPLITS, memberships)):
        raise ValueError("Duplicate episode ID within a split")
    if set.union(*memberships) != all_ids or any(memberships[i] & memberships[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Episode membership missing, extra, or overlapping")
    grouped_prompts = [{prompt_key(r) for r in splits[s] if r.get("attack_type") in {"rag", "correlated"}} for s in SPLITS]
    if any(grouped_prompts[i] & grouped_prompts[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("RAG/correlated source prompt leakage across partitions")
    all_prompts = [{prompt_key(r) for r in splits[s]} for s in SPLITS]
    return {"episode_id_overlap": 0, "rag_correlated_prompt_overlap": 0,
            "all_category_shared_prompt_counts": {f"{SPLITS[i]}_{SPLITS[j]}": len(all_prompts[i] & all_prompts[j]) for i in range(3) for j in range(i + 1, 3)},
            "scope_note": "Legacy membership and class ratios preserved. Prompt grouping is enforced for RAG/correlated only; other shared prompts are reported, not claimed absent."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--source-dir", type=Path, help="Read enriched episode files here, with original raw hash verification")
    parser.add_argument("--require-outcomes", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--membership-from", type=Path)
    parser.add_argument("--fresh-split", action="store_true")
    parser.add_argument("--benign-ratios", nargs=3, type=float, default=[0.6, 0.2, 0.2])
    parser.add_argument("--attack-ratios", nargs=3, type=float, default=[0.5, 0.2, 0.3])
    args = parser.parse_args(argv)
    if args.fresh_split and args.membership_from:
        parser.error("--fresh-split and --membership-from are mutually exclusive")
    for ratios in (args.benign_ratios, args.attack_ratios):
        if any(r < 0 or r > 1 for r in ratios) or abs(sum(ratios) - 1) > 1e-9:
            parser.error("Split ratios must be in [0, 1] and sum to one")
    by_type, hashes, source_hashes = {}, {}, {}
    source_dir = args.source_dir or args.data_dir/"raw"
    source_manifest = None
    if args.require_outcomes:
        source_manifest = json.loads((source_dir/"manifest.json").read_text())
    for file_name, attack_type in SOURCES.items():
        path = source_dir / file_name
        rows = read_jsonl(path)
        raw_path = args.data_dir/"raw"/file_name
        hashes[file_name] = sha256(raw_path)
        source_hashes[file_name] = sha256(path)
        if source_manifest:
            if source_manifest["raw_sha256"][file_name] != hashes[file_name] or source_manifest["annotated_sha256"][file_name] != source_hashes[file_name]:
                raise ValueError("Raw or annotated source changed; rerun raw annotation")
            originals = read_jsonl(raw_path)
            if len(rows) != len(originals) or any({k:v for k,v in row.items() if k != "asr_outcome"} != original for row, original in zip(rows, originals)):
                raise ValueError("Annotated source changed original trace fields/order")
        for row in rows:
            if not isinstance(row.get("id"), str) or not row["id"]:
                raise ValueError(f"Missing episode ID in {path}")
            row["label"], row["attack_type"] = int(attack_type is not None), attack_type
            if args.require_outcomes:
                from research_runner.implementation.evaluation.asr import original_outcome
                if "asr_outcome" not in row:
                    raise ValueError("Missing source outcome")
                original_outcome(normalize_trace(row))
        by_type[attack_type] = rows
    all_rows = [r for rows in by_type.values() for r in rows]
    indexed = {r["id"]: r for r in all_rows}
    if len(indexed) != len(all_rows):
        raise ValueError("Raw episode IDs are not globally unique")
    membership = None if args.fresh_split else load_membership(args.data_dir, args.membership_from)
    if membership:
        unknown = {i for ids in membership.values() for i in ids} - indexed.keys()
        if unknown:
            raise ValueError(f"Unknown preserved episode IDs: {sorted(unknown)[:10]}")
        splits = {s: [indexed[i] for i in membership[s]] for s in SPLITS}
    else:
        splits = make_splits(by_type, args.seed, args.benign_ratios, args.attack_ratios)
    integrity = validate_splits(splits, set(indexed))
    prior_manifest = args.data_dir / "splits/split_manifest.json"
    previous = json.loads(prior_manifest.read_text()) if prior_manifest.exists() else {}
    manifest = {"preprocessing_version": VERSION, "seed": args.seed,
                "requested_ratios": {"benign": args.benign_ratios, "attack": args.attack_ratios},
                "membership_policy": "preserved" if membership else "seeded_legacy_grouping",
                "membership_archive": (str(args.membership_from) if args.membership_from
                                       else previous.get("membership_archive") if membership else None),
                "raw_sha256": hashes, "source_dir": str(source_dir), "source_sha256": source_hashes,
                "asr_outcomes_required": args.require_outcomes,
                "integrity": integrity, "splits": {}, "episode_ids": {},
                "label_note": "Split label/original_label denotes scenario, not observed injection or success. Processed label denotes visible known payload."}
    for split in SPLITS:
        normalized = [normalize_trace(r) for r in splits[split]]
        write_jsonl(args.data_dir / f"splits/{split}.jsonl", normalized)
        counts = Counter(r.get("attack_type") or "benign" for r in normalized)
        manifest["splits"][split] = {
            "episodes": len(normalized), "scenario_counts": dict(counts),
            "actual_scenario_fractions": {kind or "benign": counts[kind or "benign"] / len(rows)
                                          for kind, rows in by_type.items()},
        }
        manifest["episode_ids"][split] = [r["id"] for r in normalized]
        print(f"{split}: {len(normalized)} episodes; scenarios={dict(counts)}", flush=True)
    write_json(args.data_dir / "splits/split_manifest.json", manifest)
    print(f"Membership: {manifest['membership_policy']}; integrity={integrity}", flush=True)
    return manifest


if __name__ == "__main__":
    main()
