"""Freeze a user-supplied authorization-audit prompt against existing side inputs."""
import argparse
from datetime import datetime, timezone
import difflib
import json
from pathlib import Path
import shutil

from .run import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attachment", type=Path)
    parser.add_argument("--baseline", type=Path, default=Path("results/side_vllm_layer3/20261021_072653"))
    args = parser.parse_args()
    text = args.attachment.read_text()
    start = text.index("You are O2, the final prompt-injection auditor")
    end_marker = "Return no explanation outside the JSON."
    end = text.index(end_marker, start) + len(end_marker)
    prompt = text[start:end] + "\n"
    baseline = args.baseline.resolve()
    destination = Path("results/side_l3_prompt_ablation") / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    destination.mkdir(parents=True, exist_ok=False)
    for name in ["inputs.jsonl", "requirements.lock.txt"]:
        shutil.copy2(baseline / name, destination / name)
    (destination / "system_prompt.txt").write_text(prompt)
    previous = (baseline / "system_prompt.txt").read_text()
    (destination / "prompt.diff").write_text("".join(difflib.unified_diff(previous.splitlines(True), prompt.splitlines(True), fromfile="baseline", tofile="candidate")))
    config = json.loads((baseline / "config.json").read_text())
    config.update(protocol="paired_exploratory_L3_authorization_prompt_ablation",
                  baseline_run=str(baseline), schema_variant="authorization_audit",
                  system_prompt_sha256=digest(destination / "system_prompt.txt"),
                  input_sha256=digest(destination / "inputs.jsonl"),
                  attachment_sha256=digest(args.attachment),
                  changes=["user supplied authorization audit prompt", "matching nine-field output schema"],
                  limitations=["exploratory inspected test, not validation prompt selection",
                               "same old raw-max ProtectAI L2 routing, not new detector survey",
                               "same original O2 clipping; no full-history improvement",
                               "schema differs alongside prompt; not prompt-text-only causal isolation",
                               "reporting confidence bands are not calibrated probabilities"])
    (destination / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
