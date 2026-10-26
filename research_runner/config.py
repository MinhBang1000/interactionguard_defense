"""CPU-only loading and validation of the runner's single YAML configuration.

Credentials are references to environment variable *names*, never interpolated.
Model settings and experiment parameters stay in this file, including inline
prompts and JSON schemas. No model framework is imported here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """An invalid or unsafe research configuration."""


DEFAULT_CONFIG = {
    "version": 1,
    "primary_order": "Screen-Gate-Audit",
    "naming": {"version": 2, "legacy": {"L1": "Gate", "L2": "Screen", "L3": "Audit"},
               "paper": {"Screen": "L1", "Gate": "L2", "Audit": "L3"}},
    "paths": {"workspace": ".", "results": "results/research_runner", "backup": "backup"},
    "data": {"root": "data", "judges": None, "reviews": None,
             "reference_test_sha256": None,
             "judge": {"backend": "vllm", "base_url": None, "skip": False}},
    "runtime": {"python": sys.executable, "audit_python": sys.executable,
                "device": "cpu", "max_gpu_memory_mib": 1024, "gpu_wait_seconds": 60},
    "training": {"seeds": [42], "calibration": {"gate_recall_floor": 0.999,
                                                "screen_fpr_budget": 0.01,
                                                "ae_selection_recall_floor": 0.98},
                 "excluded_attack_types": ["correlated"]},
    "models": {},
    "experiments": {},
    "backup": {"enabled": True, "include_results": True,
               "include_checkpoints": False, "include_effective_config": True,
               "checkpoint_paths": [], "exclude": [], "include_paths": [],
               "exclude_patterns": []},
}

_EXPERIMENTS = {"configuration", "baselines", "channel_ablation", "backend",
                "threshold", "benign", "timing", "auditor_targeted"}
_PATH_KEYS = {"path", "checkpoint", "checkpoint_path", "checkpoint_dir",
              "encoder_path", "autoencoder_path", "ae_path", "model_path",
              "output_dir", "cache_dir", "train_path", "validation_path", "test_path"}
_SECRET = re.compile(r"(?:^|_)(?:api_key|access_key|secret|password|credential|"
                     r"credentials|authorization|private_key|access_token|auth_token|token)(?:$|_)", re.I)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_secret_key(key: str) -> bool:
    """Whether a setting contains credentials rather than an env-name reference."""
    return bool(_SECRET.search(key)) and not key.endswith("_env")


class _UniqueLoader(yaml.SafeLoader):
    pass


def _mapping(loader: _UniqueLoader, node: yaml.MappingNode, deep: bool = False) -> dict:
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ConfigError("YAML mapping keys must be strings")
        if key in result:
            raise ConfigError(f"Duplicate YAML key: {key}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def _keys(value: Any, allowed: set[str], label: str) -> None:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    unknown = set(value) - allowed
    if unknown:
        raise ConfigError(f"Unknown {label} setting(s): {', '.join(sorted(unknown))}")


def _number(value: Any, label: str, low: float = 0, high: float | None = None) -> None:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ConfigError(f"{label} must be a finite number")
    if value < low or (high is not None and value > high):
        raise ConfigError(f"{label} is out of range [{low}, {high}]")


def _integer(value: Any, label: str, low: int = 0) -> None:
    if type(value) is not int or value < low:
        raise ConfigError(f"{label} must be an integer >= {low}")


def _text(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigError(f"{label} must be a nonempty string")


def _tree(value: Any, label: str, ancestors: frozenset[int] = frozenset()) -> None:
    """Reject cycles, non-JSON YAML objects, interpolation, and literal secrets."""
    if isinstance(value, (dict, list)):
        if id(value) in ancestors:
            raise ConfigError(f"Recursive YAML alias at {label}")
        ancestors = ancestors | {id(value)}
        entries = value.items() if isinstance(value, dict) else enumerate(value)
        for key, item in entries:
            child = f"{label}.{key}"
            if isinstance(value, dict):
                if not isinstance(key, str):
                    raise ConfigError(f"{label} keys must be strings")
                in_schema = label.endswith(".schema") or ".schema." in label
                if not in_schema and is_secret_key(key) and item not in (None, ""):
                    raise ConfigError(f"{child}: use a *_env environment variable name, not a secret")
                if not in_schema and key.endswith("_env"):
                    if not isinstance(item, str) or not _ENV_NAME.fullmatch(item):
                        raise ConfigError(f"{child} must name an environment variable")
            _tree(item, child, ancestors)
    elif isinstance(value, str):
        if "${" in value or "\x00" in value:
            raise ConfigError(f"{label}: interpolation and NUL bytes are not allowed")
    elif value is not None and type(value) not in (bool, int, float):
        raise ConfigError(f"{label} must contain only JSON-compatible values")
    elif type(value) is float and not math.isfinite(value):
        raise ConfigError(f"{label} must be finite")


def _settings(value: Any, label: str) -> None:
    """Validate common parameters without constraining inline prompts/schemas."""
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    for key, item in value.items():
        child = f"{label}.{key}"
        # JSON schemas and model-specific params are opaque JSON, checked by _tree.
        if key in {"schema", "system_prompt", "params"}:
            if key == "schema" and not isinstance(item, dict):
                raise ConfigError(f"{child} must be a mapping")
            if key == "system_prompt":
                _text(item, child)
            continue
        if key in {"resume", "enabled", "overwrite", "skip", "channel_wise",
                   "include_checkpoints", "include_results", "include_effective_config",
                   "gradient_checkpointing", "include_validation_max_f1",
                   "deduplicate_accuracy_requests", "measure_uncached"}:
            if type(item) is not bool:
                raise ConfigError(f"{child} must be a boolean")
        elif key in {"threshold", "threshold_reference", "gate_recall_floor", "screen_fpr_budget",
                     "ae_selection_recall_floor", "gpu_memory_utilization", "fpr_budget",
                     "dropout", "train_ratio", "warmup_ratio", "scheduler_factor"}:
            _number(item, child, 0, 1)
        elif key in {"seed", "sampling_seed", "max_retries", "window_overlap", "warmup", "warmup_requests"}:
            _integer(item, child)
        elif key in {"max_tokens", "max_model_len", "max_num_seqs", "max_num_batched_tokens",
                     "max_prompt_chars", "max_block_chars", "max_blocks_per_channel",
                     "workers", "concurrency", "repeats", "batch_size", "epochs", "window_size",
                     "max_length", "input_dim", "latent_dim", "gradient_accumulation",
                     "patience", "scheduler_patience", "early_stopping_patience", "total_size",
                     "max_ppl_tokens", "min_words_per_doc"}:
            _integer(item, child, 1)
        elif key in {"temperature", "learning_rate", "weight_decay", "min_delta", "max_grad_norm", "alpha"}:
            _number(item, child)
        elif key == "terminal_check":
            if not isinstance(item, str) or item not in {"binary", "Binary"}:
                raise ConfigError(f"{child} must be binary; terminal layers do not fail open")
        elif key == "terminal_policy":
            if item != "binary_at_own_threshold":
                raise ConfigError(f"{child} must be binary_at_own_threshold")
        elif key == "stage":
            if not isinstance(item, str) or item not in ("stage1", "stage2", "full"):
                raise ConfigError(f"{child} must select stage1, stage2, or full")
        elif key == "order":
            _settings({"orders": [item]}, label)
        elif key in {"ratios", "recall_floors"}:
            if not isinstance(item, list) or not item:
                raise ConfigError(f"{child} must be a nonempty list")
            for number in item:
                _number(number, child, 0, 1)
        elif key == "orders":
            if not isinstance(item, list) or not item:
                raise ConfigError(f"{child} must be a nonempty list")
            for order in item:
                if not isinstance(order, str):
                    raise ConfigError(f"{child} entries must be hyphen-separated layer names")
                layers = order.split("-")
                if not layers or len(layers) != len(set(layers)) or not set(layers) <= {"Gate", "Screen", "Audit"}:
                    raise ConfigError(f"Invalid layer order: {order}")
                if "Audit" in layers and layers[-1] != "Audit":
                    raise ConfigError(f"Audit must be terminal: {order}")
            if len(item) != len(set(item)):
                raise ConfigError(f"{child} contains duplicates")
        elif isinstance(item, dict):
            _settings(item, child)


def _merge(default: dict, supplied: dict) -> dict:
    result = copy.deepcopy(default)
    for key, value in supplied.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _path(value: Any, workspace: Path, label: str) -> str:
    _text(value, label)
    path = Path(value).expanduser()
    return str((path if path.is_absolute() else workspace / path).resolve())


def _model_paths(value: Any, workspace: Path) -> Any:
    if isinstance(value, list):
        return [_model_paths(item, workspace) for item in value]
    if not isinstance(value, dict):
        return value
    return {key: (_path(item, workspace, key) if key in _PATH_KEYS and item is not None
                  else item if key in {"schema", "params", "system_prompt"}
                  else _model_paths(item, workspace)) for key, item in value.items()}


def load_config(path: str | Path, root: str | Path | None = None) -> dict:
    """Load one YAML file; resolve paths against its configured workspace.

    ``root`` overrides the workspace. Otherwise ``paths.workspace`` is relative
    to the YAML's directory. Returned values remain JSON serializable.
    Missing optional sections receive defaults; unknown structural keys fail.
    """
    base = Path(root).expanduser().resolve() if root is not None else Path(__file__).resolve().parents[1]
    source = Path(path).expanduser()
    if not source.is_absolute():
        source = base / source
    try:
        source_bytes = source.read_bytes()
        supplied = yaml.load(source_bytes, Loader=_UniqueLoader)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Cannot load configuration {source}: {exc}") from exc
    _keys(supplied, set(DEFAULT_CONFIG), "config")
    _tree(supplied, "config")
    if type(supplied.get("version")) is not int or supplied["version"] != 1:
        raise ConfigError("version must be the integer 1")
    for section in ("paths", "data", "runtime", "training", "backup"):
        if section in supplied:
            _keys(supplied[section], set(DEFAULT_CONFIG[section]), section)
    config = _merge(DEFAULT_CONFIG, supplied)
    _settings({'order': config['primary_order']}, 'primary_order')
    if config['naming'] != DEFAULT_CONFIG['naming']:
        raise ConfigError('naming must use version 2 with the fixed legacy and paper mappings')
    _keys(config["data"]["judge"], {"backend", "base_url", "skip", "api_key_env"}, "data.judge")
    _keys(config["training"]["calibration"], {"gate_recall_floor", "screen_fpr_budget", "ae_selection_recall_floor"}, "training.calibration")
    _keys(config["models"], {"encoder", "autoencoder", "screen", "audit"}, "models")
    _keys(config["experiments"], _EXPERIMENTS, "experiments")
    for name, settings in config["experiments"].items():
        _settings(settings, f"experiments.{name}")
        if name in {'baselines', 'threshold'}:
            if 'order' in settings or 'orders' not in settings:
                raise ConfigError(f'experiments.{name}.orders is required; use a list, not order')
            if config['primary_order'] not in settings['orders']:
                raise ConfigError(f'experiments.{name}.orders must include primary_order')
        if name == 'threshold':
            from research_runner.threshold_sweep import validate_settings
            try:
                validate_settings(settings)
            except ValueError as exc:
                raise ConfigError(f'Invalid Gate sweep: {exc}') from exc
        if name == 'backend':
            if settings.get('order') != 'Screen-Gate-Audit' or 'orders' in settings:
                raise ConfigError('experiments.backend.order must be Screen-Gate-Audit; sweep candidates, not orders')
            if settings['order'] != config['primary_order']:
                raise ConfigError('experiments.backend.order must match primary_order')
            candidates = settings.get('candidates')
            if not isinstance(candidates, list) or not candidates:
                raise ConfigError('experiments.backend.candidates must be a nonempty list')
            names = []
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    raise ConfigError('Each backend candidate must be a mapping')
                _settings(candidate, 'backend candidate')
                from research_runner.modern_audit import validate_options
                try:
                    validate_options(candidate)
                except ValueError as exc:
                    raise ConfigError(f'Invalid backend candidate: {exc}') from exc
                candidate_name = candidate.get('name')
                if not isinstance(candidate_name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', candidate_name):
                    raise ConfigError('Backend candidate name must be a safe nonempty path component')
                names.append(candidate_name)
            if len(names) != len(set(names)):
                raise ConfigError('Backend candidate names must be unique')
        if name == 'benign':
            if 'order' in settings and 'orders' in settings:
                raise ConfigError('experiments.benign must specify order or orders, not both')
            if 'orders' not in settings:
                if 'order' not in settings:
                    raise ConfigError('experiments.benign.orders is required')
                settings['orders'] = [settings.pop('order')]
        if name == "configuration" and settings.get("terminal_policy") != "binary_at_own_threshold":
            raise ConfigError("experiments.configuration.terminal_policy must be binary_at_own_threshold")
        if name != "timing":
            settings.setdefault("resume", False)
    for name, settings in config["models"].items():
        if isinstance(settings, str):
            _text(settings, f"models.{name}")
        elif isinstance(settings, dict):
            _settings(settings, f"models.{name}")
        else:
            raise ConfigError(f"models.{name} must be a model identifier or mapping")
        if isinstance(settings, dict):
            for key in ("system_prompt", "schema"):
                expected = settings.get(f"{key}_sha256")
                if expected is None:
                    continue
                if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", expected):
                    raise ConfigError(f"models.{name}.{key}_sha256 must be a SHA-256 digest")
                if key not in settings:
                    raise ConfigError(f"models.{name}.{key}_sha256 has no inline {key}")
                value = settings[key]
                content = (value.encode("utf-8") if key == "system_prompt" else
                           json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False,
                                      separators=(",", ":")).encode("utf-8"))
                if hashlib.sha256(content).hexdigest() != expected.lower():
                    raise ConfigError(f"models.{name}.{key}_sha256 does not match the inline content")
    _settings(config["data"]["judge"], "data.judge")
    _settings(config["training"]["calibration"], "training.calibration")
    seeds = config["training"]["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ConfigError("training.seeds must be a nonempty list")
    for seed in seeds:
        _integer(seed, "training.seeds")
    if len(set(seeds)) != len(seeds):
        raise ConfigError("training.seeds contains duplicates")
    excluded = config["training"]["excluded_attack_types"]
    if not isinstance(excluded, list) or any(not isinstance(item, str) for item in excluded):
        raise ConfigError("training.excluded_attack_types must be a string list")
    sha = config["data"]["reference_test_sha256"]
    if sha is not None and (not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", sha)):
        raise ConfigError("data.reference_test_sha256 must be a SHA-256 hex digest or null")
    _text(config["data"]["judge"]["backend"], "data.judge.backend")
    url = config["data"]["judge"]["base_url"]
    if url is not None:
        _text(url, "data.judge.base_url")
    device = config["runtime"]["device"]
    if device is None:
        device = config["runtime"]["device"] = "cpu"
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::[0-9]+)?|[0-9]+", device):
        raise ConfigError("runtime.device must be cpu, cuda, cuda:N, or a GPU index string")
    memory = config["runtime"]["max_gpu_memory_mib"]
    if memory is None:
        memory = config["runtime"]["max_gpu_memory_mib"] = DEFAULT_CONFIG["runtime"]["max_gpu_memory_mib"]
    _integer(memory, "runtime.max_gpu_memory_mib", 1)
    _number(config["runtime"]["gpu_wait_seconds"], "runtime.gpu_wait_seconds")
    _settings(config["backup"], "backup")
    for key in ("checkpoint_paths", "exclude", "include_paths", "exclude_patterns"):
        if not isinstance(config["backup"][key], list):
            raise ConfigError(f"backup.{key} must be a list")
        for item in config["backup"][key]:
            _text(item, f"backup.{key}")
    _text(config["paths"]["workspace"], "paths.workspace")
    workspace = base if root is not None else Path(_path(config["paths"]["workspace"], source.parent, "paths.workspace"))
    config["paths"] = {key: _path(value, workspace, f"paths.{key}")
                       for key, value in config["paths"].items() if key != "workspace"}
    config["paths"]["workspace"] = str(workspace)
    for key in ("root", "judges", "reviews"):
        if key == "root":
            _text(config["data"][key], "data.root")
        if config["data"][key] is not None:
            config["data"][key] = _path(config["data"][key], workspace, f"data.{key}")
    for key in ("python", "audit_python"):
        command = config["runtime"][key]
        _text(command, f"runtime.{key}")
        if "/" in command or command.startswith("."):
            # Dereferencing a venv's bin/python symlink loses the venv identity.
            executable = Path(command).expanduser()
            config["runtime"][key] = os.path.abspath(executable if executable.is_absolute()
                                                      else workspace / executable)
    config["models"] = _model_paths(config["models"], workspace)
    for candidate in config['experiments'].get('backend', {}).get('candidates', []):
        if 'python' in candidate:
            executable = Path(candidate['python']).expanduser()
            candidate['python'] = os.path.abspath(executable if executable.is_absolute() else workspace / executable)
    for key in ("encoder", "autoencoder"):
        value = config["models"].get(key)
        if isinstance(value, str):
            config["models"][key] = _path(value, workspace, f"models.{key}")
    for key in ("checkpoint_paths", "include_paths"):
        config["backup"][key] = [_path(value, workspace, f"backup.{key}")
                                for value in config["backup"][key]]
    config["_config_path"] = str(source.absolute())
    config["_config_sha256"] = hashlib.sha256(source_bytes).hexdigest()
    if 'auditor_targeted' in config['experiments']:
        from research_runner.auditor_targeted import validate_settings
        try:
            validate_settings(config)
        except ValueError as exc:
            raise ConfigError(f'Invalid auditor-targeted case test: {exc}') from exc
    return config
