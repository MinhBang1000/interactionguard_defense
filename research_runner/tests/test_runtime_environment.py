"""Runtime repair regression tests: no real installation, compiler or GPU calls."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_runner import adapters, backend_setup, modern_audit, runtime_environment as runtime
from research_runner.artifacts import read_json, write_jsonl
from research_runner.config import load_config
from research_runner.execution import ProcessExecutor
from research_runner.tests.test_adapters import audit_config


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.python = self.root / 'venv/bin/python'
        self.python.parent.mkdir(parents=True)
        self.python.write_text('fixture')
        self.modern = {'structured_output_api': 'structured_outputs'}

    def test_modern_paths_are_local_and_parent_and_reference_env_unchanged(self):
        base = {'PATH': '/usr/bin', 'CUDA_HOME': '/usr', 'marker': 'unchanged'}
        old = dict(base)
        result = runtime.worker_environment(self.python, self.modern, base)
        self.assertEqual(base, old)
        self.assertEqual(result['PATH'].split(os.pathsep)[0], str(self.python.parent))
        self.assertEqual(result['CUDA_HOME'], str(self.root / 'venv/ig_cuda_toolkit'))
        self.assertEqual(result['FLASHINFER_NVCC'], str(self.root / 'venv/ig_cuda_toolkit/bin/nvcc'))
        self.assertEqual(runtime.worker_environment(self.python, {}, base), old)
        self.assertEqual(result['marker'], old['marker'])

    def make_toolkit(self):
        source = self.root / 'venv/lib/python3.11/site-packages/nvidia/cu13'
        for name in ('bin/nvcc', 'include/cuda_runtime.h', 'lib/libcudart.so.13', 'nvvm/bin/cicc', 'cccl/include/test.h'):
            file = source / name
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_text('fixture')
        return source

    def test_toolkit_layout_uses_venv_files_and_is_idempotent(self):
        source = self.make_toolkit()
        target = Path(runtime.create_toolkit_layout(self.python))
        self.assertEqual((target / 'bin/nvcc').resolve(), source / 'bin/nvcc')
        self.assertEqual((target / 'lib64/libcudart.so').resolve(), source / 'lib/libcudart.so.13')
        self.assertEqual((target / 'include/cccl').resolve(), source / 'cccl/include')
        self.assertEqual(runtime.create_toolkit_layout(self.python), str(target))
        self.assertTrue((target / 'lib/libcudart.so').is_file())

    def test_toolkit_refuses_overwrite_of_existing_foreign_file(self):
        self.make_toolkit()
        target = self.root / 'venv/ig_cuda_toolkit'
        target.mkdir()
        (target / 'bin').write_text('do not change')
        with self.assertRaises(runtime.PreflightError):
            runtime.create_toolkit_layout(self.python)
        self.assertEqual((target / 'bin').read_text(), 'do not change')

    def test_preflight_checks_executables_and_cuda_version_without_torch_import(self):
        self.make_toolkit()
        runtime.create_toolkit_layout(self.python)
        version = self.root / 'torch_version.py'
        version.write_text('cuda: str = "13.0"\n')
        dist = SimpleNamespace(locate_file=lambda path: version)
        with patch.object(runtime.shutil, 'which', side_effect=lambda name, **kwargs: '/tools/' + name), \
                patch.object(runtime.metadata, 'distribution', return_value=dist), \
                patch.object(runtime, '_version', side_effect=lambda command, env: 'release 13.0,' if 'nvcc' in command[0] else '1.13.2'):
            result = runtime.inspect_toolchain(self.python, self.modern, {'PATH': '/usr/bin'})
        self.assertEqual(result['nvcc_cuda'], '13.0')
        with patch.object(runtime.shutil, 'which', return_value=None), self.assertRaisesRegex(runtime.PreflightError, 'Missing worker build tools'):
            runtime.inspect_toolchain(self.python, self.modern)
        with patch.object(runtime.shutil, 'which', return_value='/compiler'), \
                patch.object(runtime.metadata, 'distribution', return_value=dist), \
                patch.object(runtime, '_version', return_value='release 11.5,'), \
                self.assertRaisesRegex(runtime.PreflightError, 'does not match PyTorch'):
            runtime.inspect_toolchain(self.python, self.modern)

    def test_cpu_preflight_reports_public_tool_error_without_raw_exception(self):
        config = load_config(adapters.ROOT / 'config/experiments.yaml')
        result = SimpleNamespace(returncode=1, stdout=json.dumps({
            'preflight_failed': True, 'error_type': 'PreflightError', 'message': 'Missing worker build tools: ninja'}))
        with patch.object(Path, 'is_file', return_value=True), patch.object(backend_setup.subprocess, 'run', return_value=result), \
                self.assertRaisesRegex(RuntimeError, 'Missing worker build tools: ninja'):
            backend_setup.check_candidates(config)

    def test_worker_subprocess_uses_same_fixed_environment_as_preflight(self):
        config = {'runtime': {'device': 'cpu', 'python': str(self.python), 'audit_python': str(self.python)},
                  'models': {'audit': {}}, 'paths': {'workspace': str(self.root)}}
        env = {}
        def child(command, *args, **kwargs):
            env.update(kwargs['env'])
            (Path(command[-1]).parent / 'worker_result.json').write_text('{}')
        with patch('research_runner.execution.logged_process', side_effect=child):
            ProcessExecutor(config).run('infer', {'backend': self.modern}, self.root / 'worker', llm=True)
        self.assertEqual(env['PATH'].split(os.pathsep)[0], str(self.python.parent))
        self.assertEqual(env['CUDA_HOME'], str(self.root / 'venv/ig_cuda_toolkit'))
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], '')

    def test_fatal_initialization_loads_once_and_never_advances_batches(self):
        config = {**audit_config(), **self.modern, 'batch_size': 1}
        backend = adapters.RealBackend({'models': {'audit': config}})
        requests = [{'request_id': str(i), 'payload': 'synthetic'} for i in range(35)]
        with patch.object(backend, '_vllm', side_effect=FileNotFoundError('ninja')) as load, \
                patch.object(backend, '_infer_vllm_batch') as batch, patch.object(modern_audit, 'emit') as emit:
            with self.assertRaisesRegex(RuntimeError, 'engine initialization failed') as caught:
                backend.infer_audit(requests)
        self.assertIsInstance(caught.exception.__cause__, FileNotFoundError)
        self.assertEqual(load.call_count, 1)
        batch.assert_not_called()
        self.assertFalse(any(c.kwargs.get('status') == 'completed' for c in emit.call_args_list))

    def test_failed_batch_stops_without_false_completed_counter(self):
        config = {**audit_config(), **self.modern, 'batch_size': 1}
        backend = adapters.RealBackend({'models': {'audit': config}})
        requests = [{'request_id': str(i), 'payload': 'synthetic'} for i in range(3)]
        values = [[{'request_id': '0', 'prediction': 0, 'error': None}],
                  [{'request_id': '1', 'prediction': None, 'error': {'type': 'RuntimeError'}}]]
        with patch.object(backend, '_vllm', return_value=(None, None)), \
                patch.object(backend, '_infer_vllm_batch', side_effect=values) as batch, \
                patch.object(modern_audit, 'emit') as emit:
            with self.assertRaisesRegex(RuntimeError, 'Audit batch failed'):
                backend.infer_audit(requests)
        self.assertEqual(batch.call_count, 2)
        self.assertTrue(any(c.kwargs.get('status') == 'failed' and c.args[1] == 1 for c in emit.call_args_list))
        self.assertFalse(any(c.kwargs.get('status') == 'completed' for c in emit.call_args_list))

    def test_gpu_smoke_orchestration_writes_only_diagnostics_and_skips_llama(self):
        config = deepcopy(load_config(adapters.ROOT / 'config/experiments.yaml'))
        config['paths']['results'] = str(self.root / 'results')
        calls = []
        def fake_worker(operation, job, destination, **kwargs):
            calls.append(job['backend']['name'])
            write_jsonl(Path(destination) / 'responses.jsonl', [{'prediction': 0, 'error': None}])
        with patch.object(backend_setup, 'check_candidates', return_value=[]), \
                patch.object(ProcessExecutor, 'run', side_effect=fake_worker), redirect_stdout(io.StringIO()):
            report = backend_setup.smoke(config)
        self.assertEqual(calls, ['qwen35_9b', 'gemma4_12b_qat_w4a16', 'nemotron3_nano_4b'])
        self.assertFalse(report['is_full_experiment'])
        self.assertEqual(set(p.name for p in (self.root / 'results').iterdir()), {'.runtime_smoke'})
        self.assertEqual(len(read_json(Path(report['output']) / 'smoke_summary.json')), 3)


if __name__ == '__main__':
    unittest.main()
