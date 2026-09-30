"""
Shared experiment utilities.

This module intentionally contains only side-effect-free helpers for CLI input,
config loading, and JSON serialization.  Experiment and detector logic stays in
the caller modules so refactors here do not change evaluation behavior.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from utils.paths import CONFIG_DIR


def ask_with_default(prompt: str, default: str) -> str:
    """Prompt for a value while keeping CLI defaults consistent across experiments."""
    value = input(prompt).strip()
    return value or default


def ask_choice(title: str, options: List[Tuple[str, str]]) -> str:
    """Render a simple numbered menu and return a validated choice."""
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    for key, label in options:
        print(f"{key}) {label}")

    valid = {str(key) for key, _ in options}
    while True:
        choice = input("Select option: ").strip()
        if choice in valid:
            return choice
        print("Invalid choice. Please choose one of:", ", ".join(sorted(valid)))


def yes_no(prompt: str, default: bool) -> bool:
    """Ask a yes/no question without changing existing y/n semantics."""
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def expand_env_vars(value: Any) -> Any:
    """Recursively expand environment variables in loaded YAML/config payloads."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_env_vars(item) for item in value]
    return value


def load_detection_config() -> Dict[str, Any]:
    """Load detection_config.yaml with environment-variable expansion."""
    with open(CONFIG_DIR / "detection_config.yaml", "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return expand_env_vars(raw)


def json_safe(obj: Any) -> Any:
    """Convert numpy-heavy experiment results into JSON-serializable values."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """Write a JSON object using the repository's stable experiment formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(json_safe(payload), f, indent=2, ensure_ascii=False)


def read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write JSONL rows with the same numpy-safe conversion as write_json."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(json_safe(row), ensure_ascii=False) + "\n")


def layer3_variant_suffix(layer3_variant: Optional[str]) -> str:
    """Convert an optional Layer 3 variant key into a stable output suffix."""
    if not layer3_variant:
        return ""
    if str(layer3_variant).startswith("layer3"):
        return f"_{layer3_variant}"
    return f"_layer3{layer3_variant}"


def normalize_layer3_variant(layer3_variant: Optional[str]) -> Optional[str]:
    """Normalize empty/default Layer 3 selections without changing custom keys."""
    if layer3_variant in {None, "", "default"}:
        return None
    return str(layer3_variant)


def default_ollama_base_url(config: Dict[str, Any]) -> str:
    """Prefer configured Ollama endpoint, falling back to localhost."""
    params = (
        config.get("layers", {})
        .get("layer3o2", {})
        .get("params", {})
    )
    configured = str(params.get("base_url", "") or "")
    if "11434" in configured:
        return configured
    return "http://localhost:11434/v1"
