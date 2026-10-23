import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research_runner.artifacts import ExistingResultsError
from research_runner.cli import MENU, main, run_selection
from research_runner.config import load_config
from research_runner.execution import ProcessExecutor, require_idle_gpu
from research_runner.engine import rebase_json
from research_runner.payloads import build_payload, request

ROOT = Path(__file__).resolve().parents[2]


class ConsoleTests(unittest.TestCase):
    def test_menu_has_exact_items_and_q(self):
        calls, messages = [], []
        class Runner:
            def __init__(self, config):
                pass
            def run(self, item, **kwargs):
                calls.append(item)
        choices = iter([*MENU, 'Q'])
        status = main([], input_fn=lambda _: next(choices), print_fn=messages.append, runner_factory=Runner)
        self.assertEqual(status, 0)
        self.assertEqual(calls, list(range(9)))
        self.assertTrue(any('Backup Snapshot' in s for s in messages))

    def test_overwrite_defaults_no_and_explicit_yes_only(self):
        class Runner:
            def __init__(self): self.calls = []
            def run(self, item, **kwargs):
                self.calls.append(kwargs)
                if not kwargs.get('overwrite'): raise ExistingResultsError('exists')
        runner = Runner()
        run_selection(runner, '2', input_fn=lambda _: '', print_fn=lambda _: None)
        self.assertEqual(runner.calls, [{}])
        run_selection(runner, '2', input_fn=lambda _: 'yes', print_fn=lambda _: None)
        self.assertEqual(runner.calls[-1], {'overwrite': True})

    def test_missing_prerequisite_english_failure_not_auto_run(self):
        messages = []
        class Runner:
            def __init__(self, config): pass
            def run(self, item): raise ValueError('Missing prerequisite: training. Run Experiment 1 first.')
        status = main(['--run', '2'], runner_factory=Runner, print_fn=messages.append)
        self.assertEqual(status, 1)
        self.assertIn('Run Experiment 1 first.', messages[-1])


class PayloadTests(unittest.TestCase):
    def test_frozen_prompt_hash_and_payload_equality_readonly(self):
        config = load_config(ROOT / 'config/experiments.yaml')
        l3 = config['models']['l3']
        self.assertEqual(hashlib.sha256(l3['system_prompt'].encode()).hexdigest(), l3['system_prompt_sha256'])
        path = ROOT / 'results/side_retrained_l1_deepset_l3/20261021/inputs.jsonl'
        if not path.is_file():
            self.skipTest('Local historical payloads are not distributed with the repository')
        with (ROOT / 'data/processed/test_processed.jsonl').open() as stream:
            rows = {r['id']: r for r in map(json.loads, stream)}
        compared = 0
        with path.open() as stream:
            for line in stream:
                old = json.loads(line)
                id_ = old.get('id', old.get('sample_id'))
                if id_ not in rows:
                    # Historical inputs use prefix_id in some snapshots.
                    id_ = old.get('prefix_id')
                if id_ in rows:
                    self.assertEqual(build_payload(rows[id_]['sequence'], l3['payload']), old['payload'])
                    compared += 1
        self.assertEqual(compared, len(rows))

    def test_channel_isolation_and_prompt_anchoring(self):
        sequence = '[PROMPT]\nUser task\n[SEP]\n[RETRIEVE]\nRAG attack\n[SEP]\n[TOOL: x]\nTool attack\n[SEP]\n[AGENT]\nAgent action'
        settings = {'max_prompt_chars': 1800, 'max_block_chars': 1200, 'max_blocks_per_channel': 2}
        isolated = build_payload(sequence, settings, 'retrieval')
        self.assertIn('User task', isolated)
        self.assertIn('RAG attack', isolated)
        self.assertNotIn('Tool attack', isolated)
        self.assertNotIn('Agent action', isolated)
        self.assertNotIn('User task', build_payload(sequence, settings, 'retrieval', 'none'))
        row = {'id': 'e_step1', 'original_trace_id': 'e', 'sequence': sequence}
        r = request(row, settings)
        self.assertEqual(r['request_id'], hashlib.sha256(r['payload'].encode()).hexdigest())


class ExecutionTests(unittest.TestCase):
    def test_publish_rebases_paths_without_changing_encoder_fingerprint(self):
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp) / 'staging'
            encoder = stage / 'seed_42/model/encoder'
            encoder.mkdir(parents=True)
            model_config = encoder / 'training_config.json'
            model_config.write_text(json.dumps({'best_checkpoint': str(encoder / 'checkpoints/epoch3')}))
            metadata = stage / 'checkpoints.json'
            metadata.write_text(json.dumps({'42': {'encoder_path': str(encoder)}}))
            before = model_config.read_bytes()
            rebase_json(stage, Path(temp) / 'published')
            self.assertEqual(model_config.read_bytes(), before)
            self.assertEqual(json.loads(metadata.read_text())['42']['encoder_path'],
                             str(Path(temp) / 'published/seed_42/model/encoder'))

    def test_gpu_busy_does_not_kill_jobs(self):
        config = {'runtime': {'device': '0', 'max_gpu_memory_mib': 1024, 'gpu_wait_seconds': 0}}
        commands = []
        def query(command, **kwargs): commands.append(command); return '9000\n'
        with self.assertRaisesRegex(RuntimeError, 'No foreign process'):
            require_idle_gpu(config, check_output=query)
        self.assertTrue(all(c[0] == 'nvidia-smi' for c in commands))

    def test_process_completion_is_stage_barrier_and_no_model_import(self):
        with tempfile.TemporaryDirectory() as temp:
            config = {'runtime': {'device': 'cpu', 'python': '/usr/bin/python3', 'l3_python': '/usr/bin/python3'},
                      'paths': {'workspace': temp}}
            executor = ProcessExecutor(config)
            events = []
            def child(command, **kwargs):
                spec = Path(command[-1]); operation = json.loads(spec.read_text())['operation']
                events.append(operation + '_started')
                (spec.parent / 'worker_result.json').write_text('{}')
                events.append(operation + '_completed')
            with patch('research_runner.execution.subprocess.run', side_effect=child):
                executor.run('score', {}, Path(temp)/'stage1')
                executor.run('infer', {}, Path(temp)/'stage2', llm=True)
            self.assertEqual(events, ['score_started', 'score_completed', 'infer_started', 'infer_completed'])


if __name__ == '__main__':
    unittest.main()
