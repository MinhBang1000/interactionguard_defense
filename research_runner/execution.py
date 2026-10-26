"""Process boundaries keep ML workers and LLM workers off the GPU together."""
import json
import os
from pathlib import Path
import subprocess
import time

from research_runner.progress import emit, logged_process


def physical_device(value):
    value = str(value)
    return '0' if value == 'cuda' else value.split(':', 1)[-1]


def require_idle_gpu(config, *, check_output=subprocess.check_output, sleep=time.sleep):
    runtime = config['runtime']
    if str(runtime['device']) == 'cpu':
        return
    deadline = time.monotonic() + runtime.get('gpu_wait_seconds', 60)
    while True:
        used = check_output(['nvidia-smi', '-i', physical_device(runtime['device']),
                             '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
        if int(used.strip()) <= runtime['max_gpu_memory_mib']:
            return
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU is busy. No foreign process will be stopped; retry when the configured GPU is idle.')
        emit('gpu_availability', status='waiting',
             message=f"GPU {runtime['device']}: {used.strip()} MiB used; waiting for <= {runtime['max_gpu_memory_mib']} MiB. No process will be stopped.", force=True)
        sleep(2)


class ProcessExecutor:
    """Blocking subprocess completion, not empty_cache(), is the stage barrier."""

    def __init__(self, config):
        self.config = config

    def run(self, operation, job, destination, *, llm=False):
        destination = Path(destination)
        destination.mkdir(parents=True, exist_ok=True)
        spec = destination / 'worker_job.json'
        spec.write_text(json.dumps({'operation': operation, 'config': self.config, 'job': job}, indent=2))
        backend = {**self.config.get('models', {}).get('audit', {}), **(job.get('backend') or {})}
        if not llm or backend.get('backend', 'vllm') == 'vllm':
            require_idle_gpu(self.config)
        runtime = self.config['runtime']
        python = runtime['audit_python'] if llm else runtime['python']
        if llm and (job.get('backend') or {}).get('python'):
            python = job['backend']['python']
        if not Path(python).is_file():
            raise RuntimeError(f'Configured Python interpreter is unavailable: {python}')
        print(f'Starting {operation} worker. Log: {destination / "worker.log"}', flush=True)
        device = physical_device(runtime['device'])
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='' if device == 'cpu' else device,
                           CUDA_DEVICE_ML=device, CUDA_DEVICE_TRAIN=device,
                           TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', VLLM_WORKER_MULTIPROC_METHOD='spawn')
        environment['IG_PROGRESS_DIR'] = str(destination / 'progress')
        if llm and backend.get('structured_output_api') == 'structured_outputs':
            from research_runner.runtime_environment import worker_environment
            environment = worker_environment(python, backend, environment)
        try:
            logged_process([python, '-u', '-m', 'research_runner.worker', str(spec)],
                           destination / 'worker.log', phase=f'{operation} worker',
                           cwd=self.config['paths']['workspace'], env=environment, check=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f'Worker {operation} failed (exit {exc.returncode}). See retained worker.log in the failed-run diagnostics.') from exc
        result = destination / 'worker_result.json'
        if not result.is_file():
            raise RuntimeError(f'Worker exited without a completion result: {result}')
        return json.loads(result.read_text())

    def preflight_audit(self, candidates):
        """Check every environment and local snapshot before the first GPU call."""
        from research_runner.backend_setup import check_candidates
        return check_candidates(self.config, candidates)
