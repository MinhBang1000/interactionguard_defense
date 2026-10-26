"""Explicit user-invoked environment/model preparation, never called implicitly.

The experiment calls only check_candidates: local files + package metadata, no
CUDA/model imports, network requests, package installations or downloads.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess

from research_runner.modern_audit import validate_options
from research_runner.progress import emit
from research_runner.runtime_environment import PreflightError, create_toolkit_layout, inspect_toolchain, worker_environment


INSPECT = r'''
import json, sys
from research_runner.backend_setup import inspect_candidate
from research_runner.runtime_environment import PreflightError
config = json.loads(sys.argv[1])
try:
    print(json.dumps(inspect_candidate(config)))
except Exception as error:
    print(json.dumps({"preflight_failed": True, "error_type": type(error).__name__,
                      "message": str(error) if isinstance(error, PreflightError) else "Dependency inspection failed; verify installed runtime packages."}))
    sys.exit(1)
'''


def inspect_candidate(config):
    import sys
    installed = metadata.version('vllm')
    if config.get('vllm_version') and installed != config['vllm_version']:
        raise PreflightError(f"Expected vLLM {config['vllm_version']}; found {installed}")
    for requirement in config.get('runtime_packages', []):
        name, expected = requirement.split('==')
        if metadata.version(name.split('[')[0]) != expected:
            raise PreflightError(f'Incorrect runtime package: {name}. Run --repair-runtime')
    toolchain = inspect_toolchain(sys.executable, config)
    from huggingface_hub import snapshot_download
    try:
        path = Path(config.get('path') or snapshot_download(config['model'], revision=config['revision'], local_files_only=True))
    except Exception as exc:
        raise PreflightError(f"Pinned model snapshot is not cached: {config['model']} @ {config['revision']}") from exc
    if not (path / 'config.json').is_file():
        raise PreflightError('Model config is not cached')
    files = {name for index in path.glob('*.safetensors.index.json')
             for name in json.loads(index.read_text())['weight_map'].values()}
    if not files:
        files = {p.name for p in path.glob('*.safetensors')}
    if not files or any(not (path / name).is_file() or not (path / name).stat().st_size for name in files):
        raise PreflightError('Model weight snapshot is missing/incomplete')
    if not (path / 'tokenizer_config.json').is_file():
        raise PreflightError('Model tokenizer is not cached')
    return {'model': config['model'], 'revision': config['revision'], 'snapshot': str(path),
            'vllm': installed, 'python': sys.version.split()[0], 'transformers': metadata.version('transformers'),
            'torch': metadata.version('torch'), 'huggingface_hub': metadata.version('huggingface_hub'),
            'weight_files': len(files), 'toolchain': toolchain}


def merged_candidates(config, candidates=None):
    candidates = candidates if candidates is not None else config['experiments']['backend']['candidates']
    return [{**config['models']['audit'], **candidate} for candidate in candidates]


def check_candidates(config, candidates=None):
    reports = []
    for candidate in merged_candidates(config, candidates):
        if candidate.get('reuse_configuration') is True:
            continue
        if candidate['backend'] != 'vllm':
            continue
        validate_options(candidate)
        python = candidate.get('python', config['runtime']['audit_python'])
        if not Path(python).is_file():
            raise RuntimeError(f"Backend {candidate.get('name')}: interpreter missing: {python}. "
                               "Run python -m research_runner.backend_setup --prepare first.")
        emit('backend.preflight', message=f"Checking {candidate.get('name')}: environment and cached weights (CPU only).", force=True)
        environment = worker_environment(python, candidate, {**os.environ, 'HF_HUB_OFFLINE': '1',
                                         'TRANSFORMERS_OFFLINE': '1', 'CUDA_VISIBLE_DEVICES': ''})
        process = subprocess.run([python, '-c', INSPECT, json.dumps(candidate)],
                                 cwd=config['paths']['workspace'], capture_output=True, text=True,
                                 env=environment, timeout=180)
        if process.returncode:
            message = 'Dependency inspection failed.'
            try:
                diagnostic = json.loads(process.stdout.strip().splitlines()[-1])
                if diagnostic.get('preflight_failed') and diagnostic.get('error_type') == 'PreflightError':
                    message = diagnostic['message']
            except (ValueError, IndexError, KeyError):
                pass
            raise RuntimeError(f"Backend {candidate.get('name')}: local preflight failed. "
                               f"{message} Run python -m research_runner.backend_setup --repair-runtime, then --check.")
        report = json.loads(process.stdout.strip().splitlines()[-1])
        reports.append({'name': candidate.get('name'), 'interpreter': python, **report})
    return reports


def repair_runtime(config):
    groups = {}
    for candidate in merged_candidates(config):
        if candidate.get('reuse_configuration') or candidate.get('structured_output_api') != 'structured_outputs':
            continue
        python = candidate.get('python')
        if not python or Path(python).absolute() == Path(config['runtime']['audit_python']).absolute():
            raise ValueError('Runtime repair must use a separate candidate environment, never reference Audit')
        packages = tuple(candidate.get('runtime_packages', ()))
        if not packages:
            raise ValueError('Modern backends require explicit runtime_packages in YAML')
        if python in groups and groups[python] != packages:
            raise ValueError('Candidates sharing an interpreter require the same runtime package pins')
        groups[python] = packages
    result = []
    for python, packages in groups.items():
        if not Path(python).is_file():
            raise ValueError('Isolated Python is missing; run --prepare first')
        print(f'Repairing build tools only in {Path(python).parent.parent}', flush=True)
        subprocess.run([python, '-m', 'pip', 'install', *packages], check=True)
        subprocess.run([python, '-m', 'pip', 'check'], check=True)
        result.append({'python': python, 'toolkit': create_toolkit_layout(python), 'packages': list(packages)})
    return result


def prepare(config):
    """Network/install actions happen only after the user's explicit --prepare."""
    groups = defaultdict(set)
    candidates = merged_candidates(config)
    candidates = [candidate for candidate in candidates if not candidate.get('reuse_configuration')]
    reference = Path(config['runtime']['audit_python']).absolute()
    for candidate in candidates:
        if candidate['backend'] == 'vllm' and candidate.get('python'):
            target = Path(candidate['python']).absolute()
            if target == reference or target.parent.name != 'bin' or target.name != 'python':
                raise ValueError('Preparation may create only a separate <venv>/bin/python, never the reference environment')
            groups[target].add(candidate['vllm_version'])
    if any(len(versions) != 1 for versions in groups.values()):
        raise ValueError('Candidates sharing an interpreter must pin the same vLLM version')
    for target, versions in groups.items():
        release = next(iter(versions))
        if not target.is_file():
            environment = target.parent.parent
            if environment.exists():
                raise ValueError(f'Refusing to overwrite an incomplete existing environment: {environment}')
            print(f'Creating isolated environment: {environment}', flush=True)
            # The reference Audit interpreter is Python 3.11; the ML conda env
            # is 3.10 on this server and cannot install vLLM 0.32.0.
            subprocess.run([str(reference), '-m', 'venv', str(environment)], check=True)
        print(f'Installing vLLM {release} into {target.parent.parent}; reference environment unchanged.', flush=True)
        subprocess.run([str(target), '-m', 'pip', 'install', f'vllm=={release}'], check=True)
        subprocess.run([str(target), '-m', 'pip', 'check'], check=True)
    repair_runtime(config)
    for candidate in candidates:
        if candidate['backend'] != 'vllm':
            continue
        python = candidate.get('python', config['runtime']['audit_python'])
        print(f"Caching pinned snapshot: {candidate['model']} @ {candidate['revision']}", flush=True)
        # HF_TOKEN is inherited if access is needed, never serialized or printed.
        subprocess.run([python, '-c',
                        'import sys; from huggingface_hub import snapshot_download; '
                        'print(snapshot_download(sys.argv[1], revision=sys.argv[2]))',
                        candidate['model'], candidate['revision']], check=True,
                       env={k: v for k, v in os.environ.items()
                            if k not in {'HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE'}})
    return check_candidates(config)


def smoke(config):
    """One synthetic request per fresh model; diagnostics, not experiment results."""
    import hashlib
    import uuid
    from research_runner.artifacts import read_jsonl, write_json, write_jsonl
    from research_runner.execution import ProcessExecutor
    check_candidates(config)
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    output = Path(config['paths']['results']) / '.runtime_smoke' / f'{timestamp}_{uuid.uuid4().hex[:8]}'
    payload = '[PROMPT]\nSummarize the visitor hours.\n\n[RETRIEVE]\nThe gallery welcomes visitors from 10:00 to 16:00.\n\n[TOOLS]\n<empty>\n\n[AGENT]\n<empty>'
    requests = output / 'synthetic_requests.jsonl'
    write_jsonl(requests, [{'request_id': hashlib.sha256(payload.encode()).hexdigest(), 'payload': payload}])
    write_json(output / 'effective_config.json', config)
    summary = {}
    executor = ProcessExecutor(config)
    for candidate in config['experiments']['backend']['candidates']:
        if candidate.get('reuse_configuration') or candidate['backend'] != 'vllm':
            continue
        try:
            destination = output / candidate['name']
            executor.run('infer', {'requests': str(requests), 'backend': candidate}, destination, llm=True)
            answers = read_jsonl(destination / 'responses.jsonl')
            if len(answers) != 1 or answers[0].get('error') or type(answers[0].get('prediction')) is not int:
                raise RuntimeError('Smoke returned no valid structured verdict')
            summary[candidate['name']] = {'status': 'passed', 'prediction': answers[0]['prediction'],
                                          'tokens': answers[0].get('token_counts'), 'seconds': answers[0].get('seconds')}
        except (OSError, ValueError, RuntimeError) as exc:
            summary[candidate['name']] = {'status': 'failed', 'error_type': type(exc).__name__, 'log': str(output / candidate['name'] / 'worker.log')}
        write_json(output / 'smoke_summary.json', summary)
    report = {'output': str(output), 'models': summary, 'is_full_experiment': False,
              'limitation': 'One synthetic request tests startup/kernels/schema, not all workload cases or research accuracy.'}
    print(json.dumps(report, indent=2))
    if any(value['status'] != 'passed' for value in summary.values()):
        raise RuntimeError(f'GPU smoke failed; inspect {output}/smoke_summary.json and worker logs')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('config/experiments.yaml'))
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--prepare', action='store_true', help='Install the isolated runtime and download pinned weights; no inference')
    action.add_argument('--check', action='store_true', help='CPU-only local preflight (default)')
    action.add_argument('--repair-runtime', action='store_true', help='Repair only the isolated build tools; no model download/inference')
    action.add_argument('--smoke', action='store_true', help='GPU diagnostic: one synthetic request per fresh model, never Llama')
    args = parser.parse_args()
    from research_runner.config import load_config
    config = load_config(args.config)
    try:
        if args.prepare:
            result = prepare(config)
        elif args.repair_runtime:
            result = repair_runtime(config)
        elif args.smoke:
            result = smoke(config)
        else:
            result = check_candidates(config)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        parser.exit(1, f'{exc}\nNo experiment was started. Check package installation, model access and cached revisions.\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
