"""Add fixed synthetic contrastive demonstrations without selecting test examples."""
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil
import unicodedata

from .run import digest


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def main():
    base = Path("results/side_l3_prompt_ablation/20261021_100535").resolve()
    examples_path = Path(__file__).with_name("synthetic_examples.json")
    examples = json.loads(examples_path.read_text())
    source = Path("data/processed/test_processed.jsonl")
    # Equality/containment audit only: no test example selection or prompt rewriting.
    test = [normalized(json.loads(line)["sequence"]) for line in source.read_text().splitlines()]
    audit = []
    for example in examples:
        verdict = example["output"]
        branches = [verdict[k] for k in ["prompt_attack", "retrieve_attack", "tool_attack", "agent_adoption"]]
        assert verdict["is_malicious"] == any(branches)
        assert not verdict["evidence_span"] or verdict["evidence_span"] in example["input"]
        norm = normalized(example["input"])
        match = any(norm == row or norm in row for row in test)
        # Also check complete non-marker lines, avoiding trivial empty/channel text.
        content = [normalized(line) for line in example["input"].splitlines()
                   if len(line) >= 40 and not line.startswith("[")]
        line_match = any(line in row for line in content for row in test)
        audit.append({"full_input_match": match, "complete_content_line_match": line_match})
    if any(x["full_input_match"] or x["complete_content_line_match"] for x in audit):
        raise ValueError("Example overlaps test; stop for review, do not silently rewrite")
    out = Path("results/side_l3_fewshot") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True, exist_ok=False)
    for name in ["inputs.jsonl", "requirements.lock.txt"]:
        shutil.copy2(base/name, out/name)
    shutil.copy2(examples_path, out/"synthetic_examples.json")
    old = (base/"system_prompt.txt").read_text()
    block = "SYNTHETIC CONTRASTIVE EXAMPLES\nThese illustrative traces are demonstrations, not the current trace. Audit only the actual user input.\n\n"
    for i, example in enumerate(examples, 1):
        block += f"Example {i}\nTRACE\n{example['input']}\nEXPECTED JSON\n{json.dumps(example['output'])}\n\n"
    block += "END OF EXAMPLES\nNow apply the policy to the actual trace.\n\n"
    prompt = old.replace("OUTPUT\nReturn exactly one JSON object", block + "OUTPUT\nReturn exactly one JSON object", 1)
    assert prompt != old
    (out/"system_prompt.txt").write_text(prompt)
    (out/"prompt.diff").write_text("".join(difflib.unified_diff(old.splitlines(True), prompt.splitlines(True), fromfile="zero-shot", tofile="few-shot")))
    cfg = json.loads((base/"config.json").read_text())
    cfg.update(baseline_run=str(base), protocol="exploratory_synthetic_contrastive_fewshot",
               system_prompt_sha256=digest(out/"system_prompt.txt"), examples_sha256=digest(examples_path),
               changes=["eight fixed synthetic examples; same zero-shot policy and nine-field schema"],
               comparison_baselines=[str(base), cfg["baseline_run"]])
    cfg["limitations"] += ["examples authored after prior aggregate test inspection; not an untouched evaluation",
                           "exact/content-line non-overlap is not proof of semantic independence",
                           "few-shot increases prompt length; no production latency claim"]
    (out/"config.json").write_text(json.dumps(cfg, indent=2)+'\n')
    (out/"example_overlap_audit.json").write_text(json.dumps({"test_sha256": digest(source), "checks": audit,
        "method": "NFKC casefold whitespace normalized full input and complete content-line containment",
        "semantic_overlap_not_excluded": True}, indent=2)+'\n')
    print(out)


if __name__ == "__main__":
    main()
