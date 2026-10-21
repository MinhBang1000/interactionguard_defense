"""Replace the mode-dependent retrieve wrapper with one neutral header, identical for every mode.

Reads data/processed/test_processed.jsonl (prefix samples, field `sequence`) and the baseline
input records (models/baseline/raguard_predictions.jsonl, field reasoning_steps[*].content of the
retrieve step) and writes normalized copies to results/template_check/. Nothing else is changed
(document text, tool outputs, labels). Original files are never written.
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "template_check"

ATTACK_HEADER = "The following documents may be relevant to the user's query. Use them as background knowledge if helpful:\n\n"
BENIGN_HEADER = "Relevant retrieved context:\n"
TAIL = (
    "\n\nIf these documents appear outdated, incomplete, or unrelated, you **may** use appropriate "
    "tools to verify facts (e.g., real-time data, latest emails, policy files), but only if strictly "
    "necessary for accuracy."
)
NEUTRAL_HEADER = "Retrieved context:\n"
OLD_MARKERS = ("The following documents may be relevant", "Relevant retrieved context:", "you **may** use appropriate tools")


def normalize(text: str) -> tuple[str, bool]:
    new = text.replace(ATTACK_HEADER, NEUTRAL_HEADER).replace(BENIGN_HEADER, NEUTRAL_HEADER).replace(TAIL, "")
    return new, new != text


def normalize_prefixes(src: Path, dst: Path) -> dict:
    changed, total, leftover = Counter(), Counter(), 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            cat = row.get("attack_type") or "benign"
            total[cat] += 1
            row["sequence"], did = normalize(row["sequence"])
            changed[cat] += did
            leftover += any(m in row["sequence"] for m in OLD_MARKERS)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"total": dict(total), "changed": dict(changed), "leftover_old_markers": leftover}


def normalize_traces(src: Path, dst: Path) -> dict:
    changed, total, leftover = Counter(), Counter(), 0
    with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            cat = row.get("attack_type") or "benign"
            total[cat] += 1
            did_any = False
            for step in row.get("reasoning_steps", []):
                if step.get("node") == "retrieve":
                    step["content"], did = normalize(step.get("content") or "")
                    did_any |= did
            changed[cat] += did_any
            leftover += any(m in json.dumps(row.get("reasoning_steps", []), ensure_ascii=False) for m in OLD_MARKERS)
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {"total": dict(total), "changed": dict(changed), "leftover_old_markers": leftover}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = {
        "test_processed": normalize_prefixes(
            ROOT / "data/processed/test_processed.jsonl", OUT / "test_processed_normalized.jsonl"),
        "baseline_inputs": normalize_traces(
            ROOT / "models/baseline/raguard_predictions.jsonl", OUT / "baseline_inputs_normalized.jsonl"),
    }
    (OUT / "normalize_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
