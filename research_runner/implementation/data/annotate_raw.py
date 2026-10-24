"""Raw -> annotated episode sources, before splitting; original raw stays immutable."""
import argparse
import hashlib
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from research_runner.implementation.data.data_preprocess import ROOT, normalize_trace, read_jsonl, sha256, write_json, write_jsonl
from research_runner.implementation.data.data_split import SOURCES
from research_runner.implementation.evaluation.annotate_trace_outcomes import annotation
from research_runner.implementation.evaluation.asr import CONTRACT, load_judges
from research_runner.implementation.evaluation.prompt_success_judge import reusable_record, apply_reviews, SYSTEM, PROMPT_VERSION


def annotate_sources(data_dir, output_dir, judge_path):
    from research_runner.progress import emit

    data_dir, output_dir, judge_path = Path(data_dir), Path(output_dir), Path(judge_path)
    judges = load_judges(judge_path)
    judge_hash = sha256(judge_path) if judge_path.exists() else None
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"contract": CONTRACT, "raw_sha256": {}, "annotated_sha256": {},
                "files": {}, "judge_sha256": judge_hash,
                "judge_prompt_version": PROMPT_VERSION,
                "judge_system_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "annotation_source_sha256": sha256(Path(__file__)),
                "notes": ["Only asr_outcome added to source trace; original fields preserved",
                          "A/B FAILURE; matching prior instruction plus target action SUCCESS",
                          "Missing/invalid judge or contradictory trace stays UNKNOWN; no model-input metadata"]}
    ids = set()
    with tempfile.TemporaryDirectory(prefix="annotated_episodes_", dir=output_dir.parent) as temporary:
        staging = Path(temporary)
        for filename, kind in SOURCES.items():
            source = data_dir/"raw"/filename
            rows, enriched = read_jsonl(source), []
            emit("dataset.annotate." + filename, 0, len(rows), unit="episodes")
            for trace in rows:
                if trace["id"] in ids:
                    raise ValueError("Duplicate raw episode ID")
                ids.add(trace["id"])
                clean = {k: v for k, v in trace.items() if k not in {"asr_outcome", "asr_outcome_v3"}}
                outcome = annotation(clean, kind, judges, judge_hash)
                if "asr_outcome" in trace and trace["asr_outcome"] != outcome:
                    raise ValueError("Raw already contains a different asr_outcome; do not overwrite original fields")
                result = {**trace, "asr_outcome": outcome}
                if {k:v for k,v in result.items() if k != "asr_outcome"} != {k:v for k,v in trace.items() if k != "asr_outcome"}:
                    raise ValueError("Raw content changed")
                enriched.append(result)
                emit("dataset.annotate." + filename, len(enriched), len(rows), unit="episodes")
            write_jsonl(staging/filename, enriched)
            if read_jsonl(staging/filename) != enriched:
                raise ValueError("Annotated file verification failed")
            manifest["raw_sha256"][filename] = sha256(source)
            manifest["annotated_sha256"][filename] = sha256(staging/filename)
            counts = Counter(x["asr_outcome"]["verdict"] for x in enriched)
            manifest["files"][filename] = {"episodes": len(rows), "counts": dict(counts)}
            emit("dataset.annotate." + filename, len(rows), len(rows), unit="episodes",
                 status="completed")
            print(f"Annotated {filename}: {len(rows)} {dict(counts)}", flush=True)
        manifest["unique_episodes"] = len(ids)
        write_json(staging/"manifest.json", manifest)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename in SOURCES:
            (staging/filename).replace(output_dir/filename)
        (staging/"manifest.json").replace(output_dir/"manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT/"data")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--judges", type=Path, default=ROOT/"eval/prompt_success_judgements.jsonl")
    parser.add_argument("--reviews", type=Path, default=ROOT/"eval/prompt_success_reviews.jsonl")
    parser.add_argument("--skip-judge", action="store_true", help="Use existing judgments only; missing evidence stays UNKNOWN")
    parser.add_argument("--judge-backend", choices=["vllm", "openai-compatible"], default="vllm")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    args = parser.parse_args(argv)
    output = args.output_dir or args.data_dir/"annotated"
    run = ROOT/"results/raw_asr_preparation"/datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    backup = run/"before"
    backup.mkdir()
    if args.judges.exists():
        shutil.copy2(args.judges, backup/"prompt_success_judgements.jsonl")
    if (output/"manifest.json").exists():
        shutil.copytree(output, backup/"annotated")
    model = "meta-llama/Llama-3.1-8B-Instruct" if args.judge_backend == "vllm" else "llama3.1:8b"
    cached = load_judges(args.judges)
    prompt_file = args.data_dir/"raw/prompt_injection_traces.jsonl"
    pending = [t for t in read_jsonl(prompt_file) if not reusable_record(normalize_trace(t), cached.get(t["id"], {}), model)]
    if pending and not args.skip_judge:
        interpreter = ROOT/"experiments/side_tests/vllm_layer3/.venv/bin/python" if args.judge_backend == "vllm" else Path(sys.executable)
        if not interpreter.exists():
            raise RuntimeError("vLLM judge environment missing; install it or choose --judge-backend openai-compatible")
        command = [str(interpreter), "-m", "research_runner.implementation.evaluation.prompt_success_judge", "--traces", str(prompt_file),
                   "--output", str(args.judges), "--backend", args.judge_backend, "--base-url", args.judge_base_url,
                   "--reviews", str(args.reviews)]
        print(f"Judge {len(pending)} uncached prompt episodes; accepted legacy verdicts preserved.", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
    cached = load_judges(args.judges)
    before = json.dumps(cached, sort_keys=True)
    apply_reviews([normalize_trace(t) for t in read_jsonl(prompt_file)], cached,
                  args.reviews)
    if before != json.dumps(cached, sort_keys=True):
        write_jsonl(args.judges, cached.values())
    manifest = annotate_sources(args.data_dir, output, args.judges)
    write_json(run/"report.json", {"output_dir": str(output), **manifest})
    print(f"Annotated source ready: {output}; report: {run}", flush=True)
    return {"run": str(run), "output_dir": str(output), **manifest}


if __name__ == "__main__":
    main()
