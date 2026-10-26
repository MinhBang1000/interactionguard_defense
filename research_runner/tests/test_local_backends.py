"""2026 local-backend configuration/setup/adapters, with no real model imports."""
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from research_runner import adapters as a, backend_setup as setup, modern_audit as modern
from research_runner.config import load_config
from research_runner.execution import ProcessExecutor
from research_runner.tests.test_adapters import audit_config, verdict


class LocalBackendTests(unittest.TestCase):
    def test_checked_in_four_models_and_unchanged_reference(self):
        config = load_config(a.ROOT / 'config/experiments.yaml')
        candidates = config['experiments']['backend']['candidates']
        self.assertEqual([c['name'] for c in candidates],
                         ['llama31_8b', 'qwen35_9b', 'gemma4_12b_qat_w4a16', 'nemotron3_nano_4b'])
        self.assertEqual(config['experiments']['backend']['order'], 'Screen-Gate-Audit')
        self.assertEqual(config['training']['seeds'], [42, 43, 44, 45, 46])
        self.assertEqual(config['models']['audit']['model'], a.AUDIT_MODEL)
        self.assertNotIn('chat_template_kwargs', config['models']['audit'])
        self.assertNotIn('python', candidates[0])
        self.assertTrue(candidates[0]['reuse_configuration'])
        for candidate in setup.merged_candidates(config):
            checked = a._audit_config(candidate)
            self.assertEqual(checked['max_model_len'], 8192)
            self.assertEqual(checked['max_tokens'], 512)
            self.assertEqual(checked['system_prompt_sha256'], a.SYSTEM_PROMPT_SHA256)
            if candidate['name'] != 'llama31_8b':
                self.assertTrue(Path(checked['python']).is_absolute())
                self.assertEqual(checked['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(candidates[2]['quantization'], 'compressed-tensors')

    def test_compact_json_is_opt_in_for_nemotron_only(self):
        config = load_config(a.ROOT / 'config/experiments.yaml')
        self.assertNotIn('structured_outputs_config', config['models']['audit'])
        for candidate in setup.merged_candidates(config):
            checked = a._audit_config(candidate)
            if candidate['name'] == 'nemotron3_nano_4b':
                self.assertEqual(checked['structured_outputs_config'],
                                 {'backend': 'xgrammar', 'disable_any_whitespace': True})
            else:
                self.assertNotIn('structured_outputs_config', checked)

    def test_invalid_compact_json_options_rejected(self):
        base = {**audit_config(), 'structured_output_api': 'structured_outputs'}
        for grammar in (True, {}, {'backend': 'auto', 'disable_any_whitespace': True},
                        {'backend': 'xgrammar', 'disable_any_whitespace': 'true'},
                        {'backend': 'xgrammar', 'disable_any_whitespace': True, 'reasoning_parser': 'custom'}):
            with self.subTest(grammar=grammar), self.assertRaisesRegex(ValueError, 'structured_outputs_config'):
                a._audit_config({**base, 'structured_outputs_config': grammar})
        with self.assertRaisesRegex(ValueError, 'modern API'):
            a._audit_config({**audit_config(), 'structured_outputs_config':
                             {'backend': 'xgrammar', 'disable_any_whitespace': True}})

    def test_invalid_runtime_options_rejected(self):
        for values in ({'language_model_only': 'true'}, {'chat_template_kwargs': {'enable_thinking': 'false'}},
                       {'chat_template_kwargs': {'tools': []}}, {'structured_output_api': 'typo'},
                       {'vllm_version': 'latest'}, {'quantization': 'q4_guess'},
                       {'python': ''}, {'generation_config': 'auto'}, {'reuse_configuration': 'true'}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                a._audit_config({**audit_config(), **values})
        with self.assertRaisesRegex(ValueError, 'Local runtime'):
            modern.validate_options({'backend': 'openai-compatible', 'python': '/tmp/python'})

    def test_non_thinking_passes_only_template_kwargs_no_prompt_rewrite(self):
        config = {**audit_config(), 'chat_template_kwargs': {'enable_thinking': False}}
        template = Mock(return_value='native non-thinking prefix')
        tokenizer = SimpleNamespace(apply_chat_template=template, encode=lambda *args, **kwargs: [1])
        completion = SimpleNamespace(text=json.dumps(verdict()), finish_reason='stop', token_ids=[2])
        llm = SimpleNamespace(get_tokenizer=lambda: tokenizer,
                              generate=lambda *args, **kwargs: [SimpleNamespace(outputs=[completion], prompt_token_ids=[1])])
        backend = a.RealBackend({'models': {'audit': config}})
        with patch.object(backend, '_vllm', return_value=(llm, None)):
            output = backend.infer_audit([{'request_id': 'one', 'payload': '[PROMPT]\nquestion'}])
        self.assertEqual(output[0]['prediction'], 0)
        args, kwargs = template.call_args
        self.assertEqual(args[0][0]['content'], config['prompt'])
        self.assertEqual(kwargs, {'tokenize': False, 'add_generation_prompt': True, 'enable_thinking': False})

    def test_new_api_quantization_text_only_and_frozen_generation(self):
        config = a._audit_config({**audit_config(), 'model': 'official/modern', 'revision': 'a' * 40,
                                  'structured_output_api': 'structured_outputs', 'vllm_version': '0.32.0',
                                  'quantization': 'compressed-tensors', 'language_model_only': True,
                                  'generation_config': 'vllm'})
        vllm, sampling = ModuleType('vllm'), ModuleType('vllm.sampling_params')
        vllm.LLM = Mock(return_value=object())
        vllm.SamplingParams = Mock()
        sampling.StructuredOutputsParams = Mock()
        backend = a.RealBackend({})
        with patch.dict(sys.modules, {'vllm': vllm, 'vllm.sampling_params': sampling}), \
                patch.object(a, '_snapshot', return_value='/cached/revision'), patch.object(modern, 'version', return_value='0.32.0'):
            backend._vllm(config)
            backend._vllm(config)
            self.assertEqual(vllm.LLM.call_count, 1)
            self.assertEqual(vllm.LLM.call_args.kwargs['quantization'], 'compressed-tensors')
            self.assertTrue(vllm.LLM.call_args.kwargs['language_model_only'])
            self.assertEqual(vllm.LLM.call_args.kwargs['generation_config'], 'vllm')
            self.assertNotIn('structured_outputs_config', vllm.LLM.call_args.kwargs)
            self.assertEqual(sampling.StructuredOutputsParams.call_args.kwargs['json'], config['schema'])
            self.assertIn('structured_outputs', vllm.SamplingParams.call_args.kwargs)
            with self.assertRaisesRegex(ValueError, 'different engine'):
                backend._vllm({**config, 'quantization': None})
        with patch.object(modern, 'version', return_value='0.11.2'), self.assertRaisesRegex(ValueError, 'Expected vLLM'):
            modern.check_version(config)

    def test_compact_json_forwarded_to_engine_and_part_of_engine_identity(self):
        config = a._audit_config({**audit_config(), 'structured_output_api': 'structured_outputs',
                                  'structured_outputs_config': {'backend': 'xgrammar', 'disable_any_whitespace': True}})
        vllm, sampling = ModuleType('vllm'), ModuleType('vllm.sampling_params')
        vllm.LLM, vllm.SamplingParams = Mock(return_value=object()), Mock()
        sampling.StructuredOutputsParams = Mock()
        backend = a.RealBackend({})
        with patch.dict(sys.modules, {'vllm': vllm, 'vllm.sampling_params': sampling}), \
                patch.object(a, '_snapshot', return_value='/cached/nemotron'):
            backend._vllm(config)
            self.assertEqual(vllm.LLM.call_args.kwargs['structured_outputs_config'], config['structured_outputs_config'])
            self.assertEqual(sampling.StructuredOutputsParams.call_args.kwargs, {'json': config['schema']})
            self.assertEqual(vllm.SamplingParams.call_args.kwargs['max_tokens'], 512)
            with self.assertRaisesRegex(ValueError, 'different engine'):
                backend._vllm({**config, 'structured_outputs_config':
                               {'backend': 'xgrammar', 'disable_any_whitespace': False}})

    def test_reference_still_uses_legacy_api_without_new_loader_kwargs(self):
        config = a._audit_config(audit_config())
        vllm, sampling = ModuleType('vllm'), ModuleType('vllm.sampling_params')
        vllm.LLM, vllm.SamplingParams = Mock(), Mock()
        sampling.GuidedDecodingParams = Mock()
        with patch.dict(sys.modules, {'vllm': vllm, 'vllm.sampling_params': sampling}), \
                patch.object(a, '_snapshot', return_value='/cached/reference'):
            a.RealBackend({})._vllm(config)
        self.assertIn('guided_decoding', vllm.SamplingParams.call_args.kwargs)
        self.assertNotIn('generation_config', vllm.LLM.call_args.kwargs)
        self.assertNotIn('language_model_only', vllm.LLM.call_args.kwargs)
        self.assertNotIn('structured_outputs_config', vllm.LLM.call_args.kwargs)

    def test_worker_interpreter_override_is_per_candidate_not_global(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {'runtime': {'device': 'cpu', 'python': sys.executable, 'audit_python': sys.executable},
                      'paths': {'workspace': temp}}
            calls = []
            def child(command, *args, **kwargs):
                calls.append(command[0])
                spec = Path(command[-1])
                (spec.parent / 'worker_result.json').write_text('{}')
            override = Path(temp) / 'venv/bin/python'
            override.parent.mkdir(parents=True)
            override.symlink_to(sys.executable)
            with patch('research_runner.execution.logged_process', side_effect=child):
                executor = ProcessExecutor(config)
                executor.run('infer', {'backend': {'python': str(override)}}, Path(temp) / 'new', llm=True)
                executor.run('infer', {}, Path(temp) / 'reference', llm=True)
            self.assertEqual(calls, [str(override), sys.executable])
            self.assertEqual(config['runtime']['audit_python'], sys.executable)

    def test_local_preflight_checks_all_candidates_offline_without_model_import(self):
        config = load_config(a.ROOT / 'config/experiments.yaml')
        def inspect(command, **kwargs):
            candidate = json.loads(command[-1])
            self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(kwargs['env']['HF_HUB_OFFLINE'], '1')
            self.assertNotIn('import torch', command[2])
            return SimpleNamespace(returncode=0, stdout=json.dumps({'model': candidate['model']}))
        with patch.object(Path, 'is_file', return_value=True), patch.object(setup.subprocess, 'run', side_effect=inspect) as run:
            reports = setup.check_candidates(config)
        self.assertEqual(len(reports), 3)
        self.assertEqual(run.call_count, 3)
        self.assertNotIn('llama31_8b', [r['name'] for r in reports])

    def test_missing_environment_does_not_install_download_or_start_experiment(self):
        config = load_config(a.ROOT / 'config/experiments.yaml')
        with patch.object(Path, 'is_file', return_value=False), patch.object(setup.subprocess, 'run') as run:
            with self.assertRaisesRegex(RuntimeError, '--prepare'):
                setup.check_candidates(config)
        run.assert_not_called()

    def test_explicit_prepare_installs_only_separate_environment_and_pins_downloads(self):
        config = load_config(a.ROOT / 'config/experiments.yaml')
        with patch.object(Path, 'is_file', return_value=True), patch.object(setup.subprocess, 'run') as run, \
                patch.object(setup, 'repair_runtime', return_value=[]), patch.object(setup, 'check_candidates', return_value=['checked']):
            self.assertEqual(setup.prepare(config), ['checked'])
        calls = [call.args[0] for call in run.call_args_list]
        install = [command for command in calls if 'install' in command]
        self.assertEqual(len(install), 1)
        self.assertIn('vllm==0.32.0', install[0])
        self.assertNotEqual(install[0][0], config['runtime']['audit_python'])
        downloads = [command for command in calls if '-c' in command]
        self.assertEqual(len(downloads), 3)
        self.assertTrue(all('meta-llama/Llama-3.1-8B-Instruct' not in command for command in downloads))
        self.assertTrue(all(len(command[-1]) == 40 for command in downloads))

    def test_new_venv_uses_reference_python311_not_ml_python310(self):
        config = load_config(a.ROOT / 'config/experiments.yaml')
        with patch.object(Path, 'is_file', return_value=False), patch.object(Path, 'exists', return_value=False), \
                patch.object(setup.subprocess, 'run') as run, patch.object(setup, 'repair_runtime', return_value=[]), \
                patch.object(setup, 'check_candidates', return_value=[]):
            setup.prepare(config)
        create = run.call_args_list[0].args[0]
        self.assertEqual(create[0], config['runtime']['audit_python'])
        self.assertIn('venv', create)
        self.assertNotEqual(create[0], config['runtime']['python'])


class LocalUpgradeTests(unittest.TestCase):
    def setUp(self):
        from research_runner.tests import test_multiconfig_upgrade as fixtures
        self.fixture = fixtures.UpgradeTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.root, self.backup = self.fixture.root, self.fixture.backup

    def test_preserves_completed_results_and_rejects_scientific_changes(self):
        from tools import upgrade_local_backends as upgrade
        from research_runner.artifacts import IncompatibleResultsError
        from research_runner.engine import ResearchRunner
        report = upgrade.apply(self.root, self.backup)
        self.assertEqual(report['status'], 'dry_run')
        report = upgrade.apply(self.root, self.backup, dry_run=False)
        self.assertEqual(report['status'], 'applied')
        before = self.fixture.before
        after = self.fixture.fixture.snapshot_artifacts()
        self.assertTrue(all(after[p] == digest for p, digest in before.items()))
        self.fixture.runner.compatible('configuration_stage1', ('dataset', 'training'))
        altered = dict(self.fixture.fixture.config)
        altered['models'] = {**altered['models'], 'audit': {**altered['models']['audit'], 'temperature': .1}}
        with self.assertRaises(IncompatibleResultsError):
            ResearchRunner(altered).compatible('configuration_stage2', ('configuration_stage1',))
        # Even an otherwise small routing edit cannot be attested.
        source = self.root / 'research_runner/engine.py'
        source.write_text(source.read_text().replace("prediction=answer['prediction']", 'prediction=0'))
        with self.assertRaisesRegex(ValueError, 'Existing scientific behavior'):
            upgrade.apply(self.root, self.backup)

    def test_threshold_scope_preserves_results_but_rejects_changes_to_other_experiments(self):
        from copy import deepcopy
        from tools import upgrade_local_backends as upgrade
        changed = deepcopy(self.fixture.original_config)
        changed['experiments']['threshold']['recall_floors'] = [.85, .95, .999]
        self.fixture.save_config(changed)
        report = upgrade.apply(self.root, self.backup, scope='threshold', dry_run=False)
        self.assertEqual(report['scope'], 'threshold')
        self.assertTrue((self.backup / 'threshold_sweep_upgrade.json').is_file())
        after = self.fixture.fixture.snapshot_artifacts()
        self.assertTrue(all(after[p] == digest for p, digest in self.fixture.before.items()))
        changed['experiments']['backend']['candidates'][0]['revision'] = 'a' * 40
        self.fixture.save_config(changed)
        with self.assertRaisesRegex(ValueError, 'Scientific inputs outside threshold'):
            upgrade.apply(self.root, self.backup, scope='threshold')


if __name__ == '__main__':
    unittest.main()
