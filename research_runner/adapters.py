"""Lazy production adapters. Importing this module never imports a GPU library.

The orchestrator calls scoring methods in disposable model workers, and calls
``infer_audit`` only in its ``runtime.audit_python`` worker. Training starts its own
``runtime.python`` child. No adapter downloads models or promotes checkpoints.

Checkpoint metadata: encoder_path, ae_path, gate_threshold, selection_path,
encoder_sha256, ae_sha256. Terminal decisions use strict score > threshold.
Audit errors have prediction/verdict=None; callers must not turn them into benign.
"""
from __future__ import annotations

from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
from urllib import request as urllib_request
from urllib.parse import urlsplit

from research_runner.progress import emit, iterable, logged_process


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_SHA256 = "638ae8020849f9631e65d4c262ffdb9f3a18cbb64ae55633acc85b2aa22ecb14"
MODEL_SPECS = {
    "deepset": ("deepset/deberta-v3-base-injection", "80dda00d0b0d9a03917a7685e2ddbcd28e04dbb1"),
    "prompt_guard_2": ("meta-llama/Llama-Prompt-Guard-2-86M", "a8ded8e697ce7c355e395a0df51f94adb4a2fd27"),
    "sentinel_v1": ("qualifire/prompt-injection-sentinel", "e24131782819a4787c129ed2b9a4a67ae57a584e"),
    "sentinel_v2": ("rogue-security/prompt-injection-jailbreak-sentinel-v2", "05ff9bfd1f28c5228e53121d5ee4427bb5205212"),
    "proventra": ("proventra/mdeberta-v3-base-prompt-injection", "b8a89d3096cf11a71d57b283c854f5ae2ed3df83"),
}
ENCODER_MODEL = "answerdotai/ModernBERT-base"
ENCODER_REVISION = "8949b909ec900327062f0ebf497f51aef5e6f0c8"
AUDIT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
AUDIT_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
BOOL_KEYS = ("prompt_attack", "retrieve_attack", "tool_attack", "agent_adoption", "is_malicious")
SECTION_RE = re.compile(r"^\[(PROMPT|RETRIEVE|TOOLS|TOOL:[^\]]+|AGENT|SEP|EOS)\]\s*$", re.M)
CHANNELS = {"prompt": "prompt", "retrieve": "retrieval", "retrieval": "retrieval",
            "tools": "tool", "tool": "tool"}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_hash(path):
    # Same contract as research_runner.implementation.training.gate_common.directory_hash, without numpy.
    digest = hashlib.sha256()
    for file in sorted(Path(path).glob("*")):
        if file.is_file() and file.suffix in {".json", ".safetensors"}:
            digest.update(file.name.encode())
            digest.update(_sha256(file).encode())
    return digest.hexdigest()


def _number(value, name, low=None, high=None):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"Invalid {name}")
    return value


def _integer(value, name, low=1):
    result = int(value)
    if isinstance(value, bool) or result != value or result < low:
        raise ValueError(f"Invalid {name}")
    return result


@contextmanager
def _offline():
    values = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _trim_middle(text, limit):
    text = str(text or "")
    if limit <= 0 or len(text) <= limit:
        return text
    marker = "\n[... middle truncated for Layer 3 O2 ...]\n"
    if limit <= len(marker) + 32:
        return text[:limit]
    head = int(limit * 0.65)
    tail = max(16, limit - head - len(marker))
    return text[:head].rstrip() + marker + text[-tail:].lstrip()


def build_payload(sequence, cfg=None):
    """Pure, byte-equivalent O2 payload builder with the frozen clipping caps.

    cfg may be the root config, models.audit, or the nested payload mapping.
    No model, OpenAI SDK, dotenv, or numpy import is needed.
    """
    cfg = cfg or {}
    cfg = cfg.get("models", {}).get("audit", cfg)
    cfg = cfg.get("payload", cfg)
    prompt_limit = int(cfg.get("max_prompt_chars", 1800))
    block_limit = int(cfg.get("max_block_chars", 1200))
    block_count = _integer(cfg.get("max_blocks_per_channel", 2), "max_blocks_per_channel")
    sections = {key: [] for key in ("PROMPT", "RETRIEVE", "TOOLS", "AGENT")}
    matches = list(SECTION_RE.finditer(sequence))
    for index, match in enumerate(matches):
        tag = match.group(1)
        if tag in {"SEP", "EOS"}:
            continue
        end = matches[index + 1].start() if index + 1 < len(matches) else len(sequence)
        content = sequence[match.end():end].strip()
        if content:
            sections["TOOLS" if tag.startswith("TOOL:") else tag].append(content)

    def blocks(label):
        values = sections[label]
        if not values:
            return f"[{label}]\n<empty>"
        return "\n\n".join(f"[{label} #{i}]\n{_trim_middle(value, block_limit)}"
                           for i, value in enumerate(values[:block_count], 1))

    prompt = _trim_middle("\n\n".join(sections["PROMPT"]), prompt_limit)
    return "\n\n".join([f"[PROMPT]\n{prompt or '<empty>'}", blocks("RETRIEVE"), blocks("TOOLS"), blocks("AGENT")])


def _classifier_spec(config):
    if isinstance(config, str):
        config = {"model": config}
    config = dict(config)
    name = config.get("model", config.get("repo", config.get("key", "deepset")))
    aliases = {"sentinel": "sentinel_v1", "pg2": "prompt_guard_2"}
    name = aliases.get(name, name)
    key = next((key for key, (repo, _) in MODEL_SPECS.items() if name in {key, repo}), None)
    if key is None:
        raise ValueError(f"Unsupported classifier: {name}")
    repo, revision = MODEL_SPECS[key]
    if config.get("revision", revision) != revision:
        raise ValueError(f"{key} must use pinned revision {revision}")
    for field, expected in {"dtype": "float32", "max_length": 512, "window_overlap": 64,
                            "aggregation": "mean", "policy": "json_mean"}.items():
        if config.get(field, expected) != expected:
            raise ValueError(f"Unsupported {field}; frozen classifier policy requires {expected}")
    return {**config, "key": key, "model": repo, "revision": revision}


def _snapshot(repo, revision, path=None, expected_hash=None):
    if path is None:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo, revision=revision, local_files_only=True)
    path = Path(path).expanduser().resolve()
    if not path.is_dir() or not (path / "config.json").is_file():
        raise ValueError(f"Local model snapshot missing: {path}")
    if expected_hash:
        if _directory_hash(path) != expected_hash:
            raise ValueError("Local model snapshot hash mismatch")
    elif path.name != revision:
        raise ValueError("Local snapshot must identify its pinned revision or supply checkpoint_sha256")
    return str(path)


def _load_window_scorer(spec, device):
    from research_runner.implementation.detectors.windowing import malicious_index
    from research_runner.implementation.detectors.windowing import WindowScorer
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

    path = _snapshot(spec["model"], spec["revision"], spec.get("path", spec.get("snapshot")), spec.get("checkpoint_sha256"))
    options = {"local_files_only": True, "trust_remote_code": False}
    config = AutoConfig.from_pretrained(path, **options)
    if config.num_labels != 2:
        raise ValueError("A binary released classifier is required")
    index, source = malicious_index(config.id2label, spec["key"])
    supplied = spec.get("malicious_label", spec.get("malicious_index", index))
    if type(supplied) is not int or supplied != index:
        raise ValueError("Configured malicious label disagrees with verified model label mapping")
    tokenizer = AutoTokenizer.from_pretrained(path, **options)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("No safe padding token")
        tokenizer.pad_token = tokenizer.eos_token
    if config.pad_token_id is None:
        config.pad_token_id = tokenizer.pad_token_id
    kwargs = {}
    if config.model_type in {"modernbert", "qwen3"}:
        kwargs["attn_implementation"] = "sdpa"
    if config.model_type == "modernbert":
        config.reference_compile = False
    model = AutoModelForSequenceClassification.from_pretrained(
        path, config=config, dtype=torch.float32, use_safetensors=True, **options, **kwargs).to(device).eval()
    model.requires_grad_(False)
    scorer = WindowScorer(SimpleNamespace(tokenizer=tokenizer, model=model, device=torch.device(device),
                                         max_length=512, window_overlap=64, malicious_index=index))
    scorer.model_details = {"model": spec["model"], "revision": spec["revision"],
                            "malicious_label": index, "mapping_source": source}
    return scorer


def _checkpoint(checkpoint):
    base = ROOT
    if not isinstance(checkpoint, dict):
        path = Path(checkpoint).expanduser().resolve()
        if path.is_dir():
            path = path / "autoencoder/selection_frozen.json"
        base = path.parent
        checkpoint = _read_json(path)
    checkpoint = dict(checkpoint)
    if "best" in checkpoint and "settings" in checkpoint:
        best = checkpoint["best"]
        checkpoint = {"encoder_path": checkpoint["settings"]["encoder_path"],
                      "ae_path": best["checkpoint"], "gate_threshold": best["calibration"]["threshold"],
                      "encoder_sha256": checkpoint["settings"].get("encoder_sha256"),
                      "ae_sha256": best.get("checkpoint_sha256")}
    for key in ("encoder_path", "ae_path", "gate_threshold"):
        if key not in checkpoint:
            raise ValueError(f"Checkpoint metadata requires {key}")
    for key in ("encoder_path", "ae_path"):
        path = Path(checkpoint[key]).expanduser()
        checkpoint[key] = str((base / path).resolve() if not path.is_absolute() else path.resolve())
    checkpoint["gate_threshold"] = _number(checkpoint["gate_threshold"], "gate_threshold")
    return checkpoint


def _load_gate(checkpoint, device):
    encoder, ae = Path(checkpoint["encoder_path"]), Path(checkpoint["ae_path"])
    if not encoder.is_dir() or not ae.is_file():
        raise ValueError("Gate encoder or autoencoder checkpoint missing")
    settings = _read_json(encoder / "training_config.json")
    if settings.get("trust_remote_code", False):
        raise ValueError("Remote model code is not permitted")
    if checkpoint.get("encoder_sha256") and _directory_hash(encoder) != checkpoint["encoder_sha256"]:
        raise ValueError("Encoder hash mismatch")
    if checkpoint.get("ae_sha256") and _sha256(ae) != checkpoint["ae_sha256"]:
        raise ValueError("Autoencoder hash mismatch")
    from research_runner.implementation.detectors.gate import GateDetector
    return GateDetector(str(encoder), str(ae), device, checkpoint["gate_threshold"])


def _release_gpu_cache():
    # Never initialize/import torch just to release a cache in a CPU parent.
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        torch.cuda.empty_cache()


@contextmanager
def _forward_progress(model, phase, total=None, unit="sequences"):
    """Observe existing forward batches without splitting/reordering model inputs."""
    current = 0
    emit(phase, 0, total, unit=unit, force=True)

    def completed_batch(module, args, kwargs, output):
        nonlocal current
        inputs = kwargs.get("input_ids")
        if inputs is None and args:
            inputs = args[0]
        if inputs is not None:
            current += len(inputs)
            emit(phase, current, total, unit=unit)

    register = getattr(model, "register_forward_hook", None)
    handle = register(completed_batch, with_kwargs=True) if callable(register) else None
    try:
        yield
    except BaseException:
        emit(phase, current, total, unit=unit, status="failed", force=True)
        raise
    else:
        # Cache-only calls have no forwards; their known input count is complete.
        final = current if total is None else total
        emit(phase, final, final, unit=unit, status="completed", force=True)
    finally:
        if handle is not None:
            handle.remove()


class RealBackend:
    """Worker-facing adapter; construction and pure payload building are CPU-safe."""

    def __init__(self, config):
        self.config = config
        self._scorers = {}
        self._gate = {}
        self._llm = None
        self._llm_key = None

    def _device(self):
        device = str(self.config.get("runtime", {}).get("device", "cpu"))
        # The executor maps every physical GPU selection into one visible GPU.
        return "cpu" if device == "cpu" else "cuda:0"

    def _release_ml(self):
        self._scorers.clear()
        self._gate.clear()
        gc.collect()
        _release_gpu_cache()

    def close(self):
        """Explicit worker cleanup; the process exit remains the stage barrier."""
        self._llm = None
        self._llm_key = None
        self._release_ml()

    def score_gate(self, rows, checkpoint):
        rows = list(rows)
        if not rows:
            return []
        checkpoint = _checkpoint(checkpoint)
        key = json.dumps(checkpoint, sort_keys=True)
        with _offline():
            if key not in self._gate:
                self._release_ml()
                emit("gate_model_load", 0, 1, unit="models", force=True)
                self._gate[key] = _load_gate(checkpoint, self._device())
                emit("gate_model_load", 1, 1, unit="models", status="completed", force=True)
            detector = self._gate[key]
            started = time.perf_counter()
            # Avoid legacy embedding-cache directories and cross-run cache reuse.
            with tempfile.TemporaryDirectory(prefix="research_gate_") as cache:
                with _forward_progress(getattr(detector, "embedding_model", None), "gate_embeddings", len(rows)):
                    embeddings = detector.extract_embedding([row["sequence"] for row in rows], batch_size=1, cache_dir=cache)
                emit("gate_reconstruction", 0, len(rows), unit="sequences", force=True)
                errors = detector.compute_reconstruction_errors(embeddings)
                emit("gate_reconstruction", len(rows), len(rows), unit="sequences", status="completed", force=True)
        if len(errors) != len(rows):
            raise ValueError("Gate returned a different number of scores")
        seconds = (time.perf_counter() - started) / len(rows)
        result = []
        for row, error in zip(rows, errors):
            score = _number(error, "Gate error", 0)
            prediction = int(score > checkpoint["gate_threshold"])
            result.append({"id": row["id"], "score": score, "error": score,
                           "prediction": prediction, "decision": "Stop" if prediction else "Forward",
                           "seconds": seconds, "timing_scope": "amortized_batch_excluding_model_load"})
        return result

    def score_detector(self, rows, model_config, channels=None):
        from research_runner.implementation.detectors.channel_payloads import channel_payloads
        from research_runner.implementation.detectors.windowing import aggregate_windows, unwrap_json

        spec = _classifier_spec(model_config)
        threshold = _number(spec.get("threshold", spec.get("threshold_reference", .5)), "Screen threshold", 0, 1)
        if isinstance(channels, str):
            channels = [channels]
        allowed = set(CHANNELS.values()) if channels is None else set()
        for channel in channels or []:
            if str(channel).lower() not in CHANNELS:
                raise ValueError(f"Unsupported detector channel: {channel}")
            allowed.add(CHANNELS[str(channel).lower()])
        rows = list(rows)
        if not rows:
            return []
        parts = [[(i, channel, unwrap_json(text)) for i, (channel, text) in enumerate(channel_payloads(row["sequence"]))
                  if channel in allowed] for row in rows]
        key = json.dumps(spec, sort_keys=True)
        texts = list(dict.fromkeys(text for blocks in parts for _, _, text in blocks))
        scorer = None
        cached_payloads = 0
        seconds = 0.
        if texts:
            with _offline():
                if key not in self._scorers:
                    self._release_ml()
                    emit("detector_model_load", 0, 1, message=spec["key"], unit="models", force=True)
                    self._scorers[key] = _load_window_scorer(spec, self._device())
                    emit("detector_model_load", 1, 1, message=spec["key"], unit="models", status="completed", force=True)
                scorer = self._scorers[key]
                cached_payloads = sum(text in scorer.cache for text in texts)
                started = time.perf_counter()
                model = getattr(getattr(scorer, "detector", None), "model", None)
                with _forward_progress(model, "detector_windows", unit="windows"):
                    scorer.score(texts)
                seconds = (time.perf_counter() - started) / len(rows)
        result = []
        for row, blocks in zip(rows, parts):
            details = []
            for index, channel, text in blocks:
                windows = [_number(value, "window probability", 0, 1) for value in scorer.cache[text]]
                if not windows:
                    raise ValueError("Classifier returned no windows for a nonempty payload")
                details.append({"channel": channel, "payload_index": index,
                                "score": aggregate_windows(windows, "mean"), "windows": windows,
                                "window_count": len(windows), "payload_sha256": hashlib.sha256(text.encode()).hexdigest()})
            score = max((detail["score"] for detail in details), default=0.)
            prediction = int(score > threshold)
            result.append({"id": row["id"], "score": score, "channels": details, "seconds": seconds,
                           "prediction": prediction, "decision": "Stop" if prediction else "Forward",
                           "model": spec["model"], "revision": spec["revision"],
                           "cached_payloads": cached_payloads,
                           "timing_scope": "cache_aware_amortized_batch_excluding_model_load"})
        return result

    def train(self, seed, output):
        """Train/calibrate on prepared train/val only, in a fresh Python process."""
        seed = _integer(seed, "seed", 0)
        output = Path(output).resolve()
        if output.exists() and any(output.iterdir()):
            raise ValueError("Refusing to overwrite a nonempty training output")
        prepared = self.config.get("_prepared_root", self.config.get("data", {}).get("_prepared_root"))
        if not prepared or not (Path(prepared) / "manifest.json").is_file():
            raise ValueError("data._prepared_root must name manifest-verified prepared training data")
        config = {key: self.config[key] for key in ("runtime", "data", "training", "_publish_root") if key in self.config}
        config["models"] = {key: self.config.get("models", {}).get(key, {}) for key in ("encoder", "autoencoder")}
        config.setdefault("data", {})
        config["data"] = {**config["data"], "_prepared_root": str(Path(prepared).resolve())}
        _training_recipe(config)
        output.mkdir(parents=True, exist_ok=True)
        runtime = config.get("runtime", {})
        env = os.environ.copy()
        device = str(runtime.get("device", "0"))
        if device == "cpu":
            raise ValueError("The production training recipe requires CUDA; use CPU mocks in tests")
        physical = "0" if device == "cuda" else device.split(":", 1)[-1]
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                   CUDA_VISIBLE_DEVICES=physical, CUDA_DEVICE_ML="0", CUDA_DEVICE_TRAIN="0")
        env["IG_PROGRESS_DIR"] = str(output / "progress")
        with tempfile.TemporaryDirectory(prefix="research_train_") as temporary:
            source, destination = Path(temporary) / "request.json", Path(temporary) / "result.json"
            source.write_text(json.dumps({"config": config, "seed": seed, "output": str(output)}), encoding="utf-8")
            command = [str(runtime.get("python", sys.executable)), "-u", "-m", "research_runner.adapters",
                       "--train-request", str(source), "--train-result", str(destination)]
            logged_process(command, output / "training.log", cwd=ROOT, env=env,
                           check=True, timeout=runtime.get("training_timeout_seconds"))
            return _read_json(destination)

    def infer_audit(self, requests, backend_config=None):
        """Run only in the isolated Audit worker; no implicit interpreter fallback."""
        cfg = {**self.config.get("models", {}).get("audit", {}), **(backend_config or {})}
        cfg = _audit_config(cfg)
        requests = list(requests)
        ids = [item["request_id"] for item in requests]
        if len(set(ids)) != len(ids) or any(not isinstance(item["payload"], str) for item in requests):
            raise ValueError("Audit requires unique request IDs and string payloads")
        if not requests:
            return []
        if self._gate or self._scorers:
            self._release_ml()
        if cfg["backend"] == "vllm":
            return self._infer_vllm(requests, cfg)
        return [_infer_http(item, cfg) for item in iterable(requests, "audit_requests", unit="requests")]

    def _infer_vllm(self, requests, cfg):
        if cfg.get("structured_output_api") == "structured_outputs":
            from research_runner.modern_audit import infer_modern_audit
            return infer_modern_audit(self, requests, cfg)
        batch_size = cfg["batch_size"]
        result = []
        emit("audit_requests", 0, len(requests), unit="requests", force=True)
        for start in iterable(range(0, len(requests), batch_size), "audit_batches", unit="batches"):
            result.extend(self._infer_vllm_batch(requests[start:start + batch_size], cfg))
            emit("audit_requests", len(result), len(requests), unit="requests")
        emit("audit_requests", len(result), len(requests), unit="requests", status="completed", force=True)
        return result

    def _infer_vllm_batch(self, requests, cfg):
        started = None
        try:
            with _offline():
                llm, params = self._vllm(cfg)
                started = time.perf_counter()
                tokenizer = llm.get_tokenizer()
                texts = [tokenizer.apply_chat_template(
                    [{"role": "system", "content": cfg["system_prompt"]}, {"role": "user", "content": item["payload"]}],
                    tokenize=False, add_generation_prompt=True, **cfg.get("chat_template_kwargs", {})) for item in requests]
                lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in texts]
                if any(length + cfg["max_tokens"] > cfg["max_model_len"] for length in lengths):
                    raise ValueError("Audit context budget exceeded; no silent truncation")
                outputs = list(llm.generate(texts, params, use_tqdm=False))
                if len(outputs) != len(requests):
                    raise ValueError("Audit returned a different number of outputs")
        except Exception as exc:
            seconds = 0. if started is None else (time.perf_counter() - started) / len(requests)
            return [_audit_error(item, exc, seconds) for item in requests]
        seconds = (time.perf_counter() - started) / len(requests)
        result = []
        for item, output in zip(requests, outputs):
            try:
                completion = output.outputs[0]
                result.append(_audit_result(item, completion.text, completion.finish_reason, cfg, seconds,
                                         len(output.prompt_token_ids), len(completion.token_ids)))
            except Exception as exc:
                result.append(_audit_error(item, exc, seconds))
        return result

    def _vllm(self, cfg):
        if cfg.get("vllm_version"):
            from research_runner.modern_audit import check_version
            check_version(cfg)
        if cfg.get("structured_output_api") == "structured_outputs":
            from research_runner.modern_audit import modern_vllm
            return modern_vllm(self, cfg)
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import GuidedDecodingParams

        fields = ("model", "revision", "dtype", "max_model_len", "gpu_memory_utilization", "max_num_seqs",
                  "max_num_batched_tokens", "seed")
        key = tuple(cfg[field] for field in fields) + (cfg.get("path"), cfg.get("checkpoint_sha256"))
        if self._llm is not None and key != self._llm_key:
            raise ValueError("Use a new Audit worker for a different engine configuration")
        if self._llm is None:
            emit("audit_model_load", 0, 1, unit="models", force=True)
            path = _snapshot(cfg["model"], cfg["revision"], cfg.get("path"), cfg.get("checkpoint_sha256"))
            os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
            self._llm = LLM(model=path, dtype=cfg["dtype"], tensor_parallel_size=1,
                            max_model_len=cfg["max_model_len"], gpu_memory_utilization=cfg["gpu_memory_utilization"],
                            max_num_seqs=cfg["max_num_seqs"], max_num_batched_tokens=cfg["max_num_batched_tokens"],
                            enable_prefix_caching=True, enable_chunked_prefill=True, seed=cfg["seed"])
            self._llm_key = key
            emit("audit_model_load", 1, 1, unit="models", status="completed", force=True)
        params = SamplingParams(temperature=cfg["temperature"], max_tokens=cfg["max_tokens"], seed=cfg["seed"],
                                guided_decoding=GuidedDecodingParams(json=cfg["schema"]))
        return self._llm, params

    def score_rag(self, rows, train_rows, cfg):
        self._release_ml()
        return _score_rag(rows, train_rows, cfg, self._device())


def _audit_config(config):
    config = dict(config)
    from research_runner.modern_audit import OPTION_KEYS, validate_options
    if OPTION_KEYS.intersection(config):
        validate_options(config)
    if any(key in config for key in ("api_key", "token", "authorization")):
        raise ValueError("Audit credentials must come from api_key_env, never configuration")
    prompt = config.get("prompt", config.get("system_prompt"))
    if not isinstance(prompt, str):
        raise ValueError("Audit requires an inline prompt/system_prompt string")
    digest = hashlib.sha256(prompt.encode()).hexdigest()
    if digest != SYSTEM_PROMPT_SHA256 or config.get("system_prompt_sha256", digest) != digest:
        raise ValueError("Audit prompt must exactly match the frozen authorization-audit prompt")
    if "prompt" in config and "system_prompt" in config and config["prompt"] != config["system_prompt"]:
        raise ValueError("Conflicting Audit prompt fields")
    properties = {key: {"type": "boolean"} for key in BOOL_KEYS}
    properties.update({key: {"type": "string"} for key in ("user_intent", "evidence_span", "reason")})
    properties["confidence"] = {"type": "number", "minimum": 0, "maximum": 1}
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    supplied_schema = config.get("schema")
    if not isinstance(supplied_schema, dict) or set(supplied_schema) != set(schema):
        raise ValueError("Audit requires the inline frozen authorization-audit schema")
    if (supplied_schema["properties"] != properties or supplied_schema["type"] != "object"
            or supplied_schema["additionalProperties"] is not False
            or not isinstance(supplied_schema["required"], list)
            or sorted(supplied_schema["required"]) != sorted(properties)):
        raise ValueError("Audit schema must exactly match the frozen authorization-audit schema")
    required = ("backend", "model", "revision", "dtype", "temperature", "seed", "max_tokens", "threshold")
    if config.get("backend") == "vllm":
        required += ("max_model_len", "max_num_seqs", "max_num_batched_tokens", "gpu_memory_utilization")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError("Audit inline configuration missing: " + ", ".join(missing))
    cfg = {**config, "system_prompt": prompt, "schema": supplied_schema}
    if cfg["backend"] not in {"vllm", "openai-compatible", "openai_compatible"}:
        raise ValueError(f"Unsupported Audit backend: {cfg['backend']}")
    if cfg["backend"] == "vllm":
        if not re.fullmatch(r"[a-f0-9]{40}", str(cfg["revision"])):
            raise ValueError("vLLM requires an immutable model revision")
        if cfg["model"] == AUDIT_MODEL and cfg["revision"] != AUDIT_REVISION:
            raise ValueError("The primary Llama model must use its frozen revision")
    cfg["threshold"] = _number(cfg["threshold"], "Audit threshold", 0, 1)
    cfg["temperature"] = _number(cfg["temperature"], "temperature", 0, 0)
    if cfg["backend"] == "vllm":
        cfg["gpu_memory_utilization"] = _number(cfg["gpu_memory_utilization"], "gpu_memory_utilization", .001, 1)
    for key in ("max_tokens", "max_model_len", "max_num_seqs", "max_num_batched_tokens"):
        if key not in cfg:
            continue
        cfg[key] = _integer(cfg[key], key)
    cfg["seed"] = _integer(cfg["seed"], "seed", 0)
    cfg["batch_size"] = _integer(cfg.get("batch_size", 64), "Audit batch_size")
    if cfg["dtype"] not in {"bfloat16", "float16", "float32"}:
        raise ValueError("Unsupported Audit dtype")
    if cfg["backend"] != "vllm":
        url = urlsplit(cfg.get("base_url", ""))
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("OpenAI-compatible base_url must be an HTTP(S) URL without credentials/query/fragment")
        if url.scheme != "https" and url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Nonlocal API endpoints require HTTPS")
        cfg["max_retries"] = _integer(cfg.get("max_retries", 3), "max_retries")
        cfg["timeout"] = _number(cfg.get("timeout", 120), "timeout", .001)
    return cfg


def _audit_result(item, text, finish_reason, cfg, seconds, prompt_tokens=None, completion_tokens=None):
    if finish_reason != "stop":
        raise ValueError("Audit completion did not stop normally; refusing a truncated/filtered verdict")
    verdict = json.loads(text)
    if not isinstance(verdict, dict) or set(verdict) != set(cfg["schema"]["properties"]):
        raise ValueError("Invalid Audit verdict keys")
    if any(type(verdict[key]) is not bool for key in BOOL_KEYS):
        raise ValueError("Audit verdict branches must be booleans")
    if any(not isinstance(verdict[key], str) for key in ("user_intent", "evidence_span", "reason")):
        raise ValueError("Audit verdict text fields must be strings")
    if type(verdict["confidence"]) not in {int, float}:
        raise ValueError("Audit confidence must be numeric")
    score = _number(verdict["confidence"], "Audit confidence", 0, 1)
    # Exactly the decision rule in experiments/side_tests/vllm_layer3/run.py.
    prediction = int(any(verdict[key] for key in BOOL_KEYS) and score >= cfg["threshold"])
    tokens = {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
              "total_tokens": None if prompt_tokens is None or completion_tokens is None else prompt_tokens + completion_tokens}
    return {"request_id": item["request_id"], "payload_sha256": hashlib.sha256(item["payload"].encode()).hexdigest(),
            "prediction": prediction, "score": score, "verdict": verdict, "seconds": seconds,
            "token_counts": tokens, "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
            "finish_reason": finish_reason, "error": None, "errors": [],
            "timing_scope": "amortized_batch_excluding_model_load" if cfg["backend"] == "vllm" else "request_including_retries"}


def _audit_error(item, exc, seconds):
    # Never serialize API exception bodies: they can contain credentials or input.
    message = str(exc) if type(exc) is ValueError else "Audit inference failed; no verdict produced"
    error = {"type": type(exc).__name__, "message": message}
    return {"request_id": item["request_id"], "payload_sha256": hashlib.sha256(item["payload"].encode()).hexdigest(),
            "prediction": None, "score": None, "verdict": None, "seconds": seconds,
            "token_counts": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
            "prompt_tokens": None, "completion_tokens": None, "finish_reason": None,
            "error": error, "errors": [error]}


def _infer_http(item, cfg):
    started = time.perf_counter()
    name = cfg.get("api_key_env", "OPENAI_API_KEY")
    token = os.environ.get(name) if name else None
    if name and not token:
        raise ValueError("Configured Audit API credential environment variable is missing")
    body = {"model": cfg["model"], "messages": [{"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": item["payload"]}], "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"], "seed": cfg["seed"],
            "response_format": {"type": "json_schema", "json_schema": {
                "name": "authorization_audit", "strict": True, "schema": cfg["schema"]}}}
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    endpoint = cfg["base_url"].rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    request = urllib_request.Request(endpoint, data=json.dumps(body).encode(), headers=headers, method="POST")
    for attempt in range(cfg["max_retries"]):
        emit("audit_http_attempts", attempt, cfg["max_retries"], unit="attempts", force=True)
        try:
            # Do not forward bearer credentials to a redirected host.
            opener = urllib_request.build_opener(_NoRedirect())
            with opener.open(request, timeout=cfg["timeout"]) as response:
                data = json.load(response)
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            emit("audit_http_attempts", attempt + 1, cfg["max_retries"], unit="attempts", status="completed", force=True)
            return _audit_result(item, choice["message"]["content"], choice.get("finish_reason"), cfg,
                              time.perf_counter() - started, usage.get("prompt_tokens"), usage.get("completion_tokens"))
        except Exception as exc:
            emit("audit_http_attempts", attempt + 1, cfg["max_retries"], unit="attempts",
                 status="failed" if attempt + 1 == cfg["max_retries"] else "running", force=True)
            if attempt + 1 == cfg["max_retries"]:
                return _audit_error(item, exc, time.perf_counter() - started)
            time.sleep(attempt + 1)


class _NoRedirect(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


ENCODER_RECIPE = {
    "max_length": 8192, "epochs": 3, "learning_rate": 2e-5, "batch_size": 4,
    "gradient_accumulation": 8, "weight_decay": .01, "warmup_ratio": .1,
    "max_grad_norm": 1., "early_stopping_patience": 2,
}
AE_RECIPE = {
    "max_length": 8192, "hidden_dims": [512, 256, 128], "latent_dim": 64, "dropout": .1,
    "epochs": 50, "batch_size": 128, "learning_rate": 3e-4, "weight_decay": 1e-5,
    "patience": 10, "min_delta": 1e-4, "search_learning_rates": [1e-3, 3e-4],
    "search_latent_dims": [32, 64], "search": True,
}


def _training_recipe(config):
    models = config.get("models", {})
    encoder = dict(models.get("encoder", {}))
    ae = dict(models.get("autoencoder", {}))
    fixed_encoder = {"pooling": "cls", "precision": "bf16", "gradient_checkpointing": True,
                     "attention_implementation": "sdpa", "train_ratio": 1.}
    fixed_ae = {"input_dim": 768, "scheduler_factor": .5, "scheduler_patience": 5}
    encoder_keys = set(ENCODER_RECIPE) | set(fixed_encoder) | {"model", "revision", "path", "checkpoint_sha256", "dtype"}
    if set(encoder) - encoder_keys or set(ae) - (set(AE_RECIPE) | set(fixed_ae)):
        raise ValueError("Unsupported encoder/autoencoder recipe fields")
    for recipe, fixed in ((encoder, fixed_encoder), (ae, fixed_ae)):
        for key, value in fixed.items():
            if recipe.get(key, value) != value:
                raise ValueError(f"The existing trainer requires {key}={value}")
    if encoder.get("model", ENCODER_MODEL) != ENCODER_MODEL or encoder.get("revision", ENCODER_REVISION) != ENCODER_REVISION:
        raise ValueError("Training requires the pinned ModernBERT-base model/revision")
    if encoder.get("dtype", "bfloat16") not in {"bfloat16", "bf16"}:
        raise ValueError("The encoder training recipe requires BF16")
    encoder = {**ENCODER_RECIPE, **fixed_encoder, "model": ENCODER_MODEL, "revision": ENCODER_REVISION, **encoder}
    ae = {**AE_RECIPE, **fixed_ae, **ae}
    if encoder["max_length"] != ae["max_length"]:
        raise ValueError("Encoder and autoencoder max_length must agree")
    for recipe in (encoder, ae):
        for key in ("max_length", "epochs", "batch_size"):
            recipe[key] = _integer(recipe[key], key)
        recipe["learning_rate"] = _number(recipe["learning_rate"], "learning_rate", 1e-15)
    calibration = config.get("training", {}).get("calibration", {})
    _number(calibration.get("gate_recall_floor", .98), "gate_recall_floor", 0, 1)
    recall = _number(calibration.get("ae_selection_recall_floor", calibration.get("gate_recall_floor", .98)),
                     "ae_selection_recall_floor", 0, 1)
    return encoder, ae, recall


def _train_worker(config, seed, output):
    """Bind the existing trainers to explicit configs inside this child only.

    The original pipeline main evaluates test and reads deployment defaults, so
    call its component trainers directly with test evaluation disabled instead.
    """
    from unittest.mock import patch
    import torch
    from research_runner.implementation.training import settings as training_settings
    from research_runner.implementation.data import data_preprocess, prepare_training
    from research_runner.implementation.training import autoencoder_training, gate_common, modernbert_finetune
    from research_runner.implementation.training.gate_runtime import gpu_preflight, verify_runtime

    encoder, ae, recall = _training_recipe(config)
    prepared = Path(config["data"]["_prepared_root"]).resolve()
    output = Path(output).resolve()
    if not (prepared / "manifest.json").is_file():
        raise ValueError("Prepared training manifest missing")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    gpu_preflight(visible)
    model = _snapshot(encoder["model"], encoder["revision"], encoder.get("path"), encoder.get("checkpoint_sha256"))
    encoder_settings = {"MODEL_NAME": model, "DATA_DIR": prepared / "encoder", "TRAIN_RATIO": 1., "SEED": seed,
                        "OUTPUT_DIR": output / "encoder", "LR": encoder["learning_rate"],
                        "GRAD_ACCUM": encoder["gradient_accumulation"],
                        "EARLY_STOP_PATIENCE": encoder["early_stopping_patience"]}
    for key in ("max_length", "epochs", "batch_size", "weight_decay", "warmup_ratio", "max_grad_norm"):
        encoder_settings[key.upper()] = encoder[key]
    ae_settings = {key.upper(): value for key, value in ae.items() if key != "search"}
    ae_settings.update(FINETUNED_MODEL_PATH=output / "encoder", MODEL_DIR=output / "autoencoder", SEED=seed)
    original_load = prepare_training.load_prepared
    original_hash = data_preprocess.sha256
    legacy_manifest = data_preprocess.ROOT / "data/training/manifest.json"

    def load(stage, split, **kwargs):
        kwargs["output_dir"] = prepared
        return original_load(stage, split, **kwargs)

    def provenance_hash(path):
        return original_hash(prepared / "manifest.json" if Path(path) == legacy_manifest else path)

    # All overrides are process-local, scoped, and restored even on failure.
    with patch.object(training_settings, "ModernBERTConfig", lambda: SimpleNamespace(**encoder_settings)), \
         patch.object(training_settings, "AutoEncoderConfig", lambda: SimpleNamespace(**ae_settings)), \
         patch.object(prepare_training, "load_prepared", load), \
         patch.object(data_preprocess, "sha256", provenance_hash), \
         patch.object(gate_common, "sha256", provenance_hash):
        modernbert_finetune.run(100, output_dir=output / "smoke", smoke=True, seed=seed)
        gc.collect()
        torch.cuda.empty_cache()
        modernbert_finetune.run(100, output_dir=output / "encoder", seed=seed)
        gc.collect()
        torch.cuda.empty_cache()
        autoencoder_training.run(encoder_dir=output / "encoder", output_dir=output / "autoencoder", seed=seed,
                                 recall_floor=recall, search=ae["search"], evaluate_test=False)
        verify_runtime(output)
    selection_path = output / "autoencoder/selection_frozen.json"
    selection = _read_json(selection_path)
    metadata = _checkpoint(selection)
    import numpy as np
    validation = original_load("encoder", "val", output_dir=prepared)
    errors = np.load(Path(metadata["ae_path"]).parent / "validation_errors.npy")
    operational_recall = config.get("training", {}).get("calibration", {}).get("gate_recall_floor", recall)
    operational = gate_common.calibrate([row["label"] for row in validation], errors, operational_recall)
    metadata["gate_threshold"] = operational["threshold"]
    metadata["gate_calibration"] = operational
    (output / "operational_calibration.json").write_text(json.dumps(operational, indent=2), encoding="utf-8")
    metadata.update(seed=seed, selection_path=str(selection_path), selection_source="validation_only",
                    test_used_for_selection=False, prepared_root=str(prepared),
                    prepared_manifest_sha256=_sha256(prepared / "manifest.json"),
                    encoder_recipe=encoder, autoencoder_recipe=ae)
    publish = config.get("_publish_root")
    if publish:
        metadata = _publish_paths(metadata, output, Path(publish).resolve())
    (output / "checkpoint.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def _publish_paths(value, output, publish):
    if isinstance(value, dict):
        return {key: _publish_paths(item, output, publish) for key, item in value.items()}
    if isinstance(value, list):
        return [_publish_paths(item, output, publish) for item in value]
    if isinstance(value, str):
        try:
            return str(publish / Path(value).relative_to(output))
        except ValueError:
            pass
    return value


def _rag_sample(row):
    if "reasoning_steps" in row:
        return row
    if "sequence" not in row:
        raise ValueError("RAGuard requires reasoning_steps or a serialized sequence")
    from research_runner.implementation.detectors.channel_payloads import channel_payloads
    parts = channel_payloads(row["sequence"])
    prompt = "\n\n".join(text for channel, text in parts if channel == "prompt")
    docs = [text for channel, text in parts if channel == "retrieval"]
    retrieve = "\n\n".join(f"[Doc {i}]\n{text}" for i, text in enumerate(docs, 1))
    return {**row, "prompt": prompt, "reasoning_steps": [{"node": "retrieve", "content": retrieve}]}


def _score_rag(rows, train_rows, cfg, device):
    """Original retrieval-only baseline, not a probability classifier.

    cfg.gpt2 and cfg.embedding are {model, revision, path?, checkpoint_sha256?}.
    Use cfg.thresholds (mapping) or fit on the supplied benign training rows.
    Native binary decisions are returned as score with score_kind marked binary.
    """
    cfg = dict(cfg)
    rows = list(rows)
    if not rows:
        return []
    model = cfg.get("gpt2")
    embedding = cfg.get("embedding")
    if model is None and cfg.get("lm_model"):
        model = {"model": cfg["lm_model"], "revision": cfg.get("lm_revision")}
    if embedding is None and cfg.get("embedding_model"):
        embedding = {"model": cfg["embedding_model"], "revision": cfg.get("embedding_revision")}
    use_ts = cfg.get("use_ts", True)
    if not isinstance(model, dict) or not model.get("model") or not model.get("revision"):
        raise ValueError("RAGuard requires an explicit pinned cfg.gpt2 model specification")
    if use_ts and (not isinstance(embedding, dict) or not embedding.get("model") or not embedding.get("revision")):
        raise ValueError("RAGuard TS requires an explicit pinned cfg.embedding model specification")
    for spec in [model] + ([embedding] if use_ts else []):
        if not re.fullmatch(r"[a-f0-9]{40}", spec["revision"]):
            raise ValueError("RAGuard model revisions must be immutable commit hashes")
    with _offline():
        from research_runner.implementation.baselines.raguard import RAGuardDetector
        emit("rag_model_load", 0, 1, unit="models", force=True)
        path = _snapshot(model["model"], model["revision"], model.get("path"), model.get("checkpoint_sha256"))
        embed_path = _snapshot(embedding["model"], embedding["revision"], embedding.get("path"), embedding.get("checkpoint_sha256")) if use_ts else ""
        detector = RAGuardDetector(gpt2_model_name=path, embedding_model_name=embed_path,
            alpha=_number(cfg.get("alpha", .025), "RAGuard alpha", 0, .5),
            max_ppl_tokens=_integer(cfg.get("max_ppl_tokens", 512), "max_ppl_tokens"),
            min_words_per_doc=_integer(cfg.get("min_words_per_doc", 20), "min_words_per_doc"),
            use_ts=use_ts, device=device, verbose=False)
        emit("rag_model_load", 1, 1, unit="models", status="completed", force=True)
        native_score = detector.score_doc

        def strict_score(query, doc):
            scores = native_score(query, doc)
            required = ["pd_signed", "pm"] + (["ts"] if use_ts else [])
            for key in required:
                _number(scores[key], "RAGuard " + key)
            return scores

        detector.score_doc = strict_score
        # The original wrapper catches scoring errors; count them explicitly.
        failures = []
        documents = 0
        document_phase = "rag_calibration_documents"

        def audited_score(query, doc):
            nonlocal documents
            try:
                return strict_score(query, doc)
            except Exception as exc:
                failures.append(type(exc).__name__)
                raise
            finally:
                documents += 1
                emit(document_phase, documents, unit="documents")

        detector.score_doc = audited_score
        thresholds = cfg.get("thresholds")
        if isinstance(thresholds, str):
            thresholds = _read_json(thresholds)
        if thresholds is not None:
            required = {"pd_low", "pd_high", "pm_high"} | ({"ts_high"} if use_ts else set())
            if not isinstance(thresholds, dict) or not required <= set(thresholds):
                raise ValueError("Incomplete RAGuard thresholds")
            detector.thresholds = {key: _number(value, "RAGuard threshold") for key, value in thresholds.items()}
        else:
            train = [_rag_sample(row) for row in train_rows]
            if not train:
                raise ValueError("RAGuard requires benign training rows or explicit frozen thresholds")
            emit(document_phase, 0, unit="documents", force=True)
            detector.fit(train, benign_label=0)
            emit(document_phase, documents, documents, unit="documents", status="completed", force=True)
            if failures:
                raise RuntimeError("RAGuard calibration skipped failed documents; refusing incomplete thresholds")
        outputs = []
        document_phase, documents = "rag_documents", 0
        emit(document_phase, 0, unit="documents", force=True)
        for row in iterable(rows, "rag_rows", unit="sequences"):
            started = time.perf_counter()
            before = len(failures)
            result = detector.predict_sample(_rag_sample(row), return_details=True)
            details = result.get("raguard_doc_results", [])
            errors = [detail for detail in details if str(detail.get("reason", "")).startswith("error:")]
            failed = len(failures) != before or bool(errors)
            prediction = None if failed else int(result["pred"])
            outputs.append({"id": row["id"], "score": prediction, "prediction": prediction,
                            "score_kind": "native_binary_not_probability", "channels": details,
                            "thresholds": dict(detector.thresholds), "seconds": time.perf_counter() - started,
                            "error": {"type": "RAGuardDocumentError", "message": "Document scoring failed"} if failed else None,
                            "limitations": ["retrieval_only", "short_documents_skipped_by_original_baseline"]})
        emit(document_phase, documents, documents, unit="documents", status="completed", force=True)
        return outputs


def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Isolated training entry point")
    parser.add_argument("--train-request", type=Path, required=True)
    parser.add_argument("--train-result", type=Path, required=True)
    args = parser.parse_args()
    request = _read_json(args.train_request)
    with _offline():
        result = _train_worker(request["config"], request["seed"], request["output"])
    args.train_result.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    _main()
