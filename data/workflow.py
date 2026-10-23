"""Back up, split episodes, freeze ASR outcomes, expand prefixes and prepare training."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile, ZIP_DEFLATED

from data.data_preprocess import ROOT


def backup_current():
    sources = [p for directory in (ROOT / "data", ROOT / "eval") for p in directory.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts]
    sources += [ROOT / path for path in ("main.py", "docs/MEMORY.md", "config/config.py",
                                         "training/modernbert_finetune.py", "training/autoencoder_training.py")]
    sources = [source for source in sources if source.is_file()]
    path = ROOT / "backups" / f"data_asr_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.zip"
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", ZIP_DEFLATED) as archive:
        for i, source in enumerate(sources, 1):
            archive.write(source, source.relative_to(ROOT))
            print(f"\rBackup {i}/{len(sources)} ({100*i/len(sources):.0f}%)", end="", flush=True)
    with ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("Backup CRC verification failed")
        for source in sources:
            if hashlib.sha256(archive.read(str(source.relative_to(ROOT)))).digest() != hashlib.sha256(source.read_bytes()).digest():
                raise ValueError(f"Backup content differs: {source}")
    print(f"\nVerified backup: {path}", flush=True)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("all", "split", "preprocess", "training", "asr", "validate"), default="all")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--judge-backend", choices=["vllm", "openai-compatible"], default="vllm")
    parser.add_argument("--judge-base-url", default="http://127.0.0.1:11434/v1")
    args = parser.parse_args(argv)
    backup_current()
    annotation_run = None
    if args.scope in {"all", "asr"}:
        from data.annotate_raw import main as annotate
        extra = ["--skip-judge"] if args.skip_judge else []
        annotated = annotate(["--judge-backend", args.judge_backend,
                              "--judge-base-url", args.judge_base_url] + extra)
        annotation_run = Path(annotated["run"])
    if args.scope in {"all", "split"}:
        from data.data_split import main as split
        split(["--source-dir", str(ROOT/"data/annotated"), "--require-outcomes"])
    if args.scope in {"all", "preprocess"}:
        from data.data_preprocess import main as preprocess
        tokenizer = ROOT / "models/layer1/answerdotai_ModernBERT-base_finetuned/tokenizer.json"
        extra = []
        if tokenizer.exists():
            try:
                import tokenizers
                extra = ["--tokenizer-path", str(tokenizer)]
            except ImportError:
                print("tokenizers unavailable: no token-length report; prefix data unchanged")
        preprocess(extra)
    if args.scope in {"all", "training"}:
        from data.prepare_training import main as prepare_training
        prepare_training([])
    if args.scope in {"all", "validate"}:
        subprocess.run([sys.executable, "-m", "unittest", "data.test_preprocessing", "data.test_prepared_training", "eval.test_asr", "eval.test_trace_annotation", "data.test_workflow", "data.test_raw_pipeline", "-v"], cwd=ROOT, check=True)
        from data.validate_preprocessing import main as validate
        validate([])
    if args.scope == "all":
        from eval.asr import evaluate_files
        for split_name in ["test"]:
            report = evaluate_files(ROOT / f"data/splits/{split_name}.jsonl",
                                    judges=ROOT / "eval/prompt_success_judgements.jsonl",
                                    output=annotation_run / f"{split_name}_baseline_asr_v3.json")
            for kind, stats in report["per_attack_type"].items():
                print(f"ASR {split_name}/{kind}: attempts={stats['total']} success={stats['success']} unknown={stats['unknown']} asr={stats['asr']}")
        print(f"Versioned ASR reports: {annotation_run}")
        print("Prompt judgments are cached; no encoder/AE training or defense inference called.")


if __name__ == "__main__":
    main()
