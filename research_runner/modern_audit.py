"""Opt-in support for modern local backends; reference Llama stays unchanged."""
from importlib.metadata import version
import json
import re

from research_runner.progress import emit


OPTION_KEYS = {"python", "vllm_version", "structured_output_api", "language_model_only",
               "quantization", "generation_config", "chat_template_kwargs", "reuse_configuration", "runtime_packages",
               "structured_outputs_config"}


def validate_options(config):
    if 'structured_outputs_config' in config:
        grammar = config['structured_outputs_config']
        if (config.get('structured_output_api') != 'structured_outputs'
                or not isinstance(grammar, dict)
                or set(grammar) != {'backend', 'disable_any_whitespace'}
                or grammar.get('backend') != 'xgrammar'
                or type(grammar.get('disable_any_whitespace')) is not bool):
            raise ValueError('structured_outputs_config requires the modern API, xgrammar and boolean disable_any_whitespace')
    if 'runtime_packages' in config:
        packages = config['runtime_packages']
        if (not isinstance(packages, list) or not packages
                or any(not isinstance(item, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9_.+!-]+', item) for item in packages)):
            raise ValueError('runtime_packages must contain exact package==version pins')
    if "reuse_configuration" in config and type(config["reuse_configuration"]) is not bool:
        raise ValueError("reuse_configuration must be boolean")
    if "python" in config and (not isinstance(config["python"], str) or not config["python"].strip()):
        raise ValueError("Backend python must be a nonempty interpreter path")
    if "vllm_version" in config and not re.fullmatch(r"\d+\.\d+\.\d+(?:\.post\d+)?", str(config["vllm_version"])):
        raise ValueError("vllm_version must pin an exact release")
    if config.get("structured_output_api", "guided_decoding") not in {"guided_decoding", "structured_outputs"}:
        raise ValueError("Unsupported structured_output_api")
    if "language_model_only" in config and type(config["language_model_only"]) is not bool:
        raise ValueError("language_model_only must be boolean")
    if config.get("quantization") not in (None, "compressed-tensors"):
        raise ValueError("Only explicitly supported compressed-tensors quantization is allowed")
    if "generation_config" in config and config["generation_config"] != "vllm":
        raise ValueError("Explicit generation_config must be vllm (no hidden model sampling overrides)")
    if "chat_template_kwargs" in config:
        kwargs = config["chat_template_kwargs"]
        if (not isinstance(kwargs, dict) or set(kwargs) != {"enable_thinking"}
                or type(kwargs["enable_thinking"]) is not bool):
            raise ValueError("chat_template_kwargs must contain only boolean enable_thinking")
    if config.get("backend", "vllm") != "vllm" and OPTION_KEYS.intersection(config):
        raise ValueError("Local runtime options require backend: vllm")


def check_version(config):
    expected = config.get("vllm_version")
    if expected and version("vllm") != expected:
        raise ValueError(f"Expected vLLM {expected}; prepare the candidate's isolated Python environment")


def infer_modern_audit(backend, requests, config):
    """Fatal engine/batch failures stop immediately, never 35 fake successes."""
    from research_runner.progress import iterable
    from research_runner.adapters import _offline
    emit('audit_requests', 0, len(requests), unit='requests', force=True)
    try:
        with _offline():
            backend._vllm(config)
    except Exception as exc:
        emit('audit_model_load', status='failed', message='Audit engine initialization failed; stopping before inference.', force=True)
        raise RuntimeError('Audit engine initialization failed. See worker.log; no requests were classified and no automatic batch retry was performed.') from exc
    result = []
    for start in iterable(range(0, len(requests), config['batch_size']), 'audit_batches', unit='batches'):
        selected = requests[start:start + config['batch_size']]
        batch = backend._infer_vllm_batch(selected, config)
        if ([item.get('request_id') for item in batch] != [item['request_id'] for item in selected]
                or any(item.get('error') or type(item.get('prediction')) is not int
                       or item['prediction'] not in (0, 1) for item in batch)):
            emit('audit_requests', len(result), len(requests), unit='requests', status='failed',
                 message='Invalid Audit batch; refusing to mark failed requests as completed.', force=True)
            raise RuntimeError('Audit batch failed or returned an invalid/truncated verdict. See worker.log; failed requests are not benign predictions.')
        result.extend(batch)
        emit('audit_requests', len(result), len(requests), unit='requests')
    emit('audit_requests', len(result), len(requests), unit='requests', status='completed', force=True)
    return result


def modern_vllm(backend, config):
    """Use the current structured-output API without changing the legacy branch."""
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from research_runner.adapters import _snapshot

    fields = ("model", "revision", "dtype", "max_model_len", "gpu_memory_utilization",
              "max_num_seqs", "max_num_batched_tokens", "seed", "path", "checkpoint_sha256",
              "language_model_only", "quantization", "generation_config", "structured_outputs_config")
    key = json.dumps({k: config.get(k) for k in fields}, sort_keys=True)
    if backend._llm is not None and backend._llm_key != key:
        raise ValueError("Use a new Audit worker for a different engine configuration")
    if backend._llm is None:
        emit("audit_model_load", 0, 1, unit="models", force=True)
        path = _snapshot(config["model"], config["revision"], config.get("path"), config.get("checkpoint_sha256"))
        options = {k: config[k] for k in ("language_model_only", "quantization", "generation_config",
                                          "structured_outputs_config") if k in config}
        backend._llm = LLM(
            model=path, dtype=config["dtype"], tensor_parallel_size=1,
            max_model_len=config["max_model_len"], gpu_memory_utilization=config["gpu_memory_utilization"],
            max_num_seqs=config["max_num_seqs"], max_num_batched_tokens=config["max_num_batched_tokens"],
            enable_prefix_caching=True, enable_chunked_prefill=True, seed=config["seed"], **options)
        backend._llm_key = key
        emit("audit_model_load", 1, 1, unit="models", status="completed", force=True)
    params = SamplingParams(temperature=config["temperature"], max_tokens=config["max_tokens"], seed=config["seed"],
                            structured_outputs=StructuredOutputsParams(json=config["schema"]))
    return backend._llm, params
