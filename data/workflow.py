"""Back up current data, regenerate in place, and validate the existing split."""
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
    parser.add_argument("--scope", choices=("all", "split", "preprocess", "training"), default="all")
    args = parser.parse_args(argv)
    backup_current()
    if args.scope in {"all", "split"}:
        from data.data_split import main as split
        split([])
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
    if args.scope in {"all", "preprocess", "training"}:
        from data.prepare_training import main as prepare_training
        prepare_training([])
        subprocess.run([sys.executable, "-m", "unittest", "data.test_preprocessing", "data.test_prepared_training", "eval.test_asr", "-v"], cwd=ROOT, check=True)
        from data.validate_preprocessing import main as validate
        validate([])
        from eval.asr import evaluate_files
        report = evaluate_files(ROOT / "data/splits/test.jsonl",
                                judges=ROOT / "eval/prompt_success_judgements.jsonl",
                                output=ROOT / "eval/asr_offline_no_defense.json")
        print("No API judge called. Missing/stale prompt judgments remain UNKNOWN.")
        for kind, stats in report["per_attack_type"].items():
            print(f"ASR {kind}: attempts={stats['total']} success={stats['success']} unknown={stats['unknown']} asr={stats['asr']}")


if __name__ == "__main__":
    main()
