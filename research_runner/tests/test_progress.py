"""CPU-only progress tests; subprocesses and models are stand-ins."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from research_runner import progress


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.output = io.StringIO()
        self.now = 100.
        self.reporter = progress.Reporter(self.path, stream=self.output, clock=lambda: self.now)
        token = progress._current.set(self.reporter)
        self.addCleanup(progress._current.reset, token)

    def state(self, path=None):
        return json.loads(((path or self.path) / 'progress.json').read_text())

    def test_percent_eta_and_atomic_history(self):
        self.reporter.update('batches', 0, 10, unit='batches')
        self.now += 5
        self.reporter.update('batches', 2, 10, unit='batches')
        state = self.state()
        self.assertEqual(state['percent'], 20.)
        self.assertEqual(state['eta_seconds'], 20.)
        self.assertIn('2/10 batches (20.0%)', self.output.getvalue())
        self.assertIn('ETA 00:00:20', self.output.getvalue())
        self.assertEqual(len((self.path / 'progress.jsonl').read_text().splitlines()), 2)
        self.assertFalse(list(self.path.glob('*.tmp')))

    def test_unknown_total_has_no_invented_percent_or_eta_and_is_throttled(self):
        self.reporter.update('loading', message='Loading weights')
        self.now += .1
        self.assertIsNone(self.reporter.update('loading', message='Still loading'))
        state = self.state()
        self.assertIsNone(state['percent'])
        self.assertIsNone(state['eta_seconds'])
        self.assertNotIn('%', self.output.getvalue())

    def test_reset_new_batch_and_terminal_status_not_throttled(self):
        self.reporter.update('epoch', 4, 5)
        self.now += 5
        self.reporter.update('epoch', 1, 5)
        self.assertEqual(self.state()['elapsed_seconds'], 0.)
        self.reporter.update('epoch', 1, 5, status='completed')
        self.assertEqual(self.state()['status'], 'completed')
        self.assertEqual(self.state()['percent'], 20.)  # Early stopping is not 100%.

    def test_invalid_counts_and_status_fail(self):
        for kwargs in ({'current': -1}, {'current': True}, {'total': -1},
                       {'current': 3, 'total': 2}, {'status': 'error'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.reporter.update('invalid', **kwargs)

    def test_gpu_wait_reports_memory_without_changing_foreign_processes(self):
        from research_runner.execution import require_idle_gpu
        samples = iter(['9000', '0'])
        config = {'runtime': {'device': '0', 'max_gpu_memory_mib': 1024, 'gpu_wait_seconds': 60}}
        require_idle_gpu(config, check_output=lambda *a, **kw: next(samples), sleep=lambda _: None)
        self.assertEqual(self.state()['status'], 'waiting')
        self.assertIn('9000 MiB', self.state()['message'])

    def test_overwrite_conflict_is_waiting_not_a_failed_experiment(self):
        destination = self.path / 'overwrite'
        with redirect_stdout(self.output), self.assertRaises(FileExistsError), progress.scope(destination, 'menu'):
            raise FileExistsError('Existing results')
        self.assertEqual(self.state(destination)['status'], 'waiting')

    def test_empty_and_generator_iterables_preserve_values(self):
        self.assertEqual(list(progress.iterable([], 'empty')), [])
        self.assertEqual(self.state()['status'], 'completed')
        self.assertEqual(self.state()['current'], 0)
        self.assertEqual(list(progress.iterable((i*i for i in range(3)), 'generator')), [0, 1, 4])
        self.assertEqual(self.state()['current'], 3)
        self.assertIsNone(self.state()['total'])

    def test_interrupted_iteration_does_not_claim_completion(self):
        def broken():
            yield 1
            raise RuntimeError('fixture')
        with self.assertRaisesRegex(RuntimeError, 'fixture'):
            list(progress.iterable(broken(), 'broken'))
        self.assertNotEqual(self.state()['status'], 'completed')

    def test_scope_restores_context_and_records_completion_failure_cancel(self):
        with redirect_stdout(self.output):
            for status, error in (('completed', None), ('failed', ValueError('fixture')), ('cancelled', KeyboardInterrupt())):
                destination = self.path / status
                if error is None:
                    with progress.scope(destination, 'menu'):
                        progress.emit('batch', 1, 1)
                else:
                    with self.assertRaises(type(error)), progress.scope(destination, 'menu'):
                        raise error
                self.assertEqual(self.state(destination)['status'], status)
                self.assertIs(progress.reporter(), self.reporter)

    def test_scope_heartbeat_during_silent_operation(self):
        destination = self.path / 'heartbeat'
        with redirect_stdout(self.output), progress.scope(destination, 'loading', heartbeat_seconds=.01):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if self.state(destination)['status'] == 'waiting':
                    break
                time.sleep(.005)
            else:
                self.fail('Silent operation did not emit a heartbeat')
            self.assertIsNone(self.state(destination)['percent'])
        self.assertEqual(self.state(destination)['status'], 'completed')
        self.assertFalse(any(t.name == 'interactionguard-heartbeat' for t in threading.enumerate()))

    def test_live_worker_logs_reach_console_before_process_returns(self):
        observed = []
        def child(command, **kwargs):
            log = kwargs['stdout']
            log.write('private raw trace text is not a progress line\n')
            log.write('[IG] audit_batches | running [##########----------] 1/2 batches (50.0%) | elapsed 00:00:01\n')
            log.flush()
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if '1/2 batches' in self.output.getvalue():
                    observed.append('visible_before_exit')
                    break
                time.sleep(.005)
            return subprocess.CompletedProcess(command, 0)
        with patch('research_runner.progress.subprocess.run', side_effect=child):
            progress.logged_process(['fake-worker'], self.path / 'worker.log', poll_seconds=.005)
        self.assertEqual(observed, ['visible_before_exit'])
        self.assertNotIn('private raw trace', self.output.getvalue())
        self.assertIn('private raw trace', (self.path / 'worker.log').read_text())
        records = [json.loads(line) for line in (self.path / 'progress.jsonl').read_text().splitlines()]
        self.assertTrue(any(r['current'] == 1 and r['total'] == 2 for r in records))
        self.assertEqual(self.state()['status'], 'completed')
        self.assertEqual(self.output.getvalue().count('1/2 batches'), 1)
        self.assertFalse(any(t.name == 'interactionguard-log-watch' for t in threading.enumerate()))

    def test_subprocess_failure_retains_log_and_records_failure(self):
        with patch('research_runner.progress.subprocess.run', side_effect=subprocess.CalledProcessError(2, ['fake'])):
            with self.assertRaises(subprocess.CalledProcessError):
                progress.logged_process(['fake'], self.path / 'failed.log')
        self.assertEqual(self.state()['status'], 'failed')
        self.assertTrue((self.path / 'failed.log').is_file())
        self.assertFalse(any(t.name == 'interactionguard-log-watch' for t in threading.enumerate()))

    def test_nonzero_unchecked_subprocess_is_not_marked_completed(self):
        with patch('research_runner.progress.subprocess.run', return_value=subprocess.CompletedProcess(['fake'], 7)):
            result = progress.logged_process(['fake'], self.path / 'nonzero.log')
        self.assertEqual(result.returncode, 7)
        self.assertEqual(self.state()['status'], 'failed')

    def test_subprocess_heartbeat_does_not_terminate_child(self):
        def child(command, **kwargs):
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline:
                if 'Process is active' in self.output.getvalue():
                    return subprocess.CompletedProcess(command, 0)
                time.sleep(.005)
            self.fail('Worker heartbeat missing')
        with patch('research_runner.progress.subprocess.run', side_effect=child) as run:
            progress.logged_process(['fake'], self.path / 'silent.log', heartbeat_seconds=.01, poll_seconds=.005)
        run.assert_called_once()
        self.assertEqual(self.state()['status'], 'completed')

    def test_trainer_callback_reports_steps_validation_and_early_stop(self):
        callback = progress.trainer_callback(object)
        state = SimpleNamespace(global_step=0, max_steps=20, epoch=0.)
        callback.on_train_begin(None, state, None)
        state.global_step, state.epoch = 3, .5
        self.now += 2
        callback.on_step_end(None, state, None)
        self.assertEqual(self.state()['current'], 3)
        callback.on_log(None, state, None, logs={'loss': .2})
        self.assertIn('loss', self.state()['message'])
        callback.on_prediction_step(None, state, None, eval_dataloader=[1, 2])
        callback.on_prediction_step(None, state, None, eval_dataloader=[1, 2])
        self.assertEqual(self.state()['current'], 2)
        callback.on_evaluate(None, state, None, metrics={'eval_f1': .9})
        self.assertEqual(callback.eval_batches, 0)
        callback.on_train_end(None, state, None)
        self.assertEqual(self.state()['status'], 'completed')
        self.assertEqual(self.state()['percent'], 15.)

    def test_forward_hook_observes_batches_and_is_removed_on_error(self):
        from research_runner.adapters import _forward_progress
        class Model:
            removed = False
            def register_forward_hook(model, hook, **kwargs):
                model.hook = hook
                return SimpleNamespace(remove=lambda: setattr(model, 'removed', True))
        model = Model()
        with _forward_progress(model, 'windows', total=3):
            model.hook(model, (), {'input_ids': [[1], [2]]}, None)
            model.hook(model, (), {'input_ids': [[3]]}, None)
        self.assertTrue(model.removed)
        self.assertEqual(self.state()['current'], 3)
        with self.assertRaisesRegex(RuntimeError, 'model error'), _forward_progress(model, 'failed_windows'):
            raise RuntimeError('model error')
        self.assertTrue(model.removed)
        self.assertEqual(self.state()['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
