#!/usr/bin/env python3
"""Create a timestamped project backup with a visible progress bar.

The backup is intentionally broad enough for experiment reproducibility:
source code, configs, data files, eval files, docs, and results are included.
Large model/checkpoint artifacts, caches, nested backups, Git internals, and
secret-bearing env files are excluded to keep archives portable and safe.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


ROOT_DIR = Path(__file__).resolve().parents[1]

EXCLUDED_DIR_NAMES = {
    ".git",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "backups",
    "node_modules",
    "venv",
}

EXCLUDED_FILE_NAMES = {
    ".env",
    ".env.local",
}

EXCLUDED_SUFFIXES = {
    ".arrow",
    ".bin",
    ".ckpt",
    ".ggml",
    ".gguf",
    ".joblib",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".pyc",
    ".safetensors",
    ".zip",
}


@dataclass(frozen=True)
class BackupPlan:
    files: List[Path]
    skipped_count: int
    skipped_bytes: int


def _iter_progress(items: Sequence[Path], label: str) -> Iterable[Path]:
    try:
        from tqdm.auto import tqdm

        yield from tqdm(items, desc=label, unit="file")
    except Exception:
        total = len(items)
        width = 30
        for index, item in enumerate(items, start=1):
            filled = int(width * index / max(1, total))
            bar = "#" * filled + "-" * (width - filled)
            print(f"\r{label}: [{bar}] {index}/{total}", end="", flush=True)
            yield item
        print()


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0:
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _is_backup_dir(path: Path) -> bool:
    return path.name.startswith("backup_") or path.name.startswith("backup-")


def _should_skip_dir(path: Path) -> bool:
    return path.name in EXCLUDED_DIR_NAMES or _is_backup_dir(path)


def _should_skip_file(path: Path, max_file_bytes: int) -> bool:
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    try:
        if path.stat().st_size > max_file_bytes:
            return True
    except OSError:
        return True
    return False


def build_backup_plan(root: Path, max_file_mb: int) -> BackupPlan:
    max_file_bytes = int(max_file_mb) * 1024 * 1024
    files: List[Path] = []
    skipped_count = 0
    skipped_bytes = 0

    for dirpath, dirnames, filenames in os.walk(root):
        current = Path(dirpath)
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if not _should_skip_dir(current / dirname)
        ]

        for filename in filenames:
            path = current / filename
            try:
                size = path.stat().st_size
            except OSError:
                skipped_count += 1
                continue

            if _should_skip_file(path, max_file_bytes=max_file_bytes):
                skipped_count += 1
                skipped_bytes += int(size)
                continue

            files.append(path)

    files.sort(key=lambda item: item.relative_to(root).as_posix())
    return BackupPlan(files=files, skipped_count=skipped_count, skipped_bytes=skipped_bytes)


def create_backup(label: str, backup_dir: Path, max_file_mb: int, dry_run: bool = False) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in label).strip("_")
    safe_label = safe_label or "current_results"
    archive = backup_dir / f"{safe_label}_{timestamp}.zip"

    plan = build_backup_plan(ROOT_DIR, max_file_mb=max_file_mb)
    total_bytes = sum(path.stat().st_size for path in plan.files)

    print(f"Project root     : {ROOT_DIR}")
    print(f"Backup archive   : {archive}")
    print(f"Files to include : {len(plan.files)}")
    print(f"Payload size     : {_format_bytes(total_bytes)}")
    print(f"Skipped files    : {plan.skipped_count} ({_format_bytes(plan.skipped_bytes)})")
    print("Excluded         : .git, caches, nested backups, .env, model weights/checkpoints, large files")

    if dry_run:
        print("\nDry run only. No archive was created.")
        return archive

    backup_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root_dir": str(ROOT_DIR),
        "archive": str(archive),
        "file_count": len(plan.files),
        "payload_bytes": total_bytes,
        "skipped_count": plan.skipped_count,
        "skipped_bytes": plan.skipped_bytes,
        "max_file_mb": int(max_file_mb),
        "excluded_dir_names": sorted(EXCLUDED_DIR_NAMES),
        "excluded_file_names": sorted(EXCLUDED_FILE_NAMES),
        "excluded_suffixes": sorted(EXCLUDED_SUFFIXES),
    }

    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as zf:
        zf.writestr("backup_manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for path in _iter_progress(plan.files, "Backing up"):
            zf.write(path, path.relative_to(ROOT_DIR).as_posix())

    print("\nBackup complete.")
    print(f"Archive size     : {_format_bytes(archive.stat().st_size)}")
    print("Integrity check  : running")
    with zipfile.ZipFile(archive, "r") as zf:
        bad_file = zf.testzip()
    if bad_file:
        raise RuntimeError(f"Backup integrity check failed at: {bad_file}")
    print("Integrity check  : OK")
    return archive


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", nargs="?", default="current_results")
    parser.add_argument("--backup-dir", default="backups")
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=200,
        help="Skip individual files larger than this limit. Default: 200.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    create_backup(
        label=args.label,
        backup_dir=Path(args.backup_dir),
        max_file_mb=args.max_file_mb,
        dry_run=bool(args.dry_run),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
