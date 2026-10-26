"""CPU functional tests: real Item 2 -> Item 6 reuse and safe inference fallback."""
from copy import deepcopy
from pathlib import Path
import unittest

from research_runner.artifacts import read_json, read_jsonl, write_json
from research_runner.backend_reuse import REPORT_FILES
from research_runner.tests import test_engine as fixtures
from research_runner.threshold_reuse import ThresholdReference, publish_threshold_reference


class ThresholdReuseTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EngineTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)
        self.runner, self.config = self.fixture.runner, self.fixture.config
        original = self.fixture.backend.train

        def train(seed, output):
            checkpoint = original(seed, output)
            checkpoint['gate_threshold'] = .1
            write_json(Path(output) / 'checkpoint.json', checkpoint)
            return checkpoint

        self.fixture.backend.train = train
        self.config['experiments']['threshold'].update(
            recall_floors=[.9, .999], include_validation_max_f1=False,
            reuse_configuration_recall_floors=[.999])

    def prepare(self, stage2=True):
        self.fixture.prepare(training=True)
        self.runner.configuration_stage1()
        if stage2:
            self.runner.configuration_stage2()

    def test_exact_reports_reused_for_both_orders_without_audit_calls(self):
        self.prepare()
        before = self.fixture.snapshot_artifacts()
        calls = len(self.fixture.backend.calls)
        self.runner.run(6)
        new = self.fixture.backend.calls[calls:]
        # Only recall .9 is freshly audited: 2 seeds x 2 orders.
        self.assertEqual(len(new), 4)
        self.assertTrue(all(call[0] == 'infer_audit' for call in new))
        output = self.fixture.results / 'threshold'
        for seed in self.config['training']['seeds']:
            for order in self.config['experiments']['threshold']['orders']:
                condition = output / f'seed_{seed}/recall_0.999/{order}'
                original = self.fixture.results / f'configuration_stage2/seed_{seed}/{order}'
                for name in REPORT_FILES:
                    self.assertEqual((condition / name).read_bytes(), (original / name).read_bytes())
                execution = read_json(condition / 'execution.json')
                self.assertEqual(execution['result_source'], 'inherited_item_2')
                self.assertEqual(execution['new_audit_requests'], 0)
                self.assertEqual(execution['latency_source'], 'inherited_item_2')
                self.assertTrue(all(r['operating_point'] == 'recall_0.999'
                                    for r in read_jsonl(condition / 'audit/requests.jsonl')))
        summary = read_json(output / 'summary.json')
        for order in self.config['experiments']['threshold']['orders']:
            item = summary[f'recall_0.999/{order}']
            self.assertEqual(item['new_inference_trials'], 0)
            self.assertEqual(item['execution_provenance']['reused_seeds'], [42, 43])
        self.assertTrue((output / 'execution_summary.csv').is_file())
        self.assertTrue((output / 'execution_protocol.json').is_file())
        after = self.fixture.snapshot_artifacts()
        self.assertTrue(all(after[p] == digest for p, digest in before.items()))

    def test_missing_item2_stage2_falls_back_to_fresh_inference(self):
        self.prepare(stage2=False)
        calls = len(self.fixture.backend.calls)
        self.runner.run(6)
        self.assertEqual(len(self.fixture.backend.calls) - calls, 8)
        execution = read_json(self.fixture.results / 'threshold/seed_42/recall_0.999/Screen-Gate-Audit/execution.json')
        self.assertEqual(execution['result_source'], 'new_item_6_inference')
        self.assertIn('unavailable_or_incompatible', execution['reuse_fallback_reason'])

    def reference_inputs(self, order='Screen-Gate-Audit'):
        first = self.fixture.results / 'configuration_stage1'
        rows = self.runner.rows()
        staged, pending, _, _ = self.runner._backend_inputs(first, 42, rows, order)
        pending = [{**r, 'operating_point': 'recall_0.999'} for r in pending]
        return ThresholdReference(self.runner, first), rows, staged, pending

    def test_wrong_threshold_or_payload_never_reuses(self):
        self.prepare()
        reference, rows, staged, pending = self.reference_inputs()
        calibration = {'recall_floor': .999, 'threshold': .2}
        bundle, reason = reference.lookup(42, 'Screen-Gate-Audit', rows, staged, pending, calibration)
        self.assertIsNone(bundle)
        self.assertIn('threshold differs', reason)
        altered = deepcopy(pending)
        altered[0]['payload'] = 'wrong payload'
        bundle, reason = reference.lookup(42, 'Screen-Gate-Audit', rows, staged, altered,
                                          {'recall_floor': .999, 'threshold': .1})
        self.assertIsNone(bundle)
        self.assertIn('IDs/payloads differ', reason)

    def test_model_mismatch_falls_back_instead_of_using_stale_results(self):
        self.prepare()
        self.config['models']['audit']['threshold'] = .6
        calls = len(self.fixture.backend.calls)
        self.runner.run(6)
        self.assertEqual(len(self.fixture.backend.calls) - calls, 8)
        execution = read_json(self.fixture.results / 'threshold/seed_42/recall_0.999/Screen-Gate-Audit/execution.json')
        self.assertEqual(execution['result_source'], 'new_item_6_inference')
        self.assertIn('incompatible', execution['reuse_fallback_reason'])

    def test_source_changed_after_validation_is_not_published(self):
        self.prepare()
        reference, rows, staged, pending = self.reference_inputs()
        bundle, reason = reference.lookup(42, 'Screen-Gate-Audit', rows, staged, pending,
                                          {'recall_floor': .999, 'threshold': .1})
        self.assertIsNone(reason)
        (bundle['directory'] / 'metrics.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'changed after validation'):
            publish_threshold_reference(bundle, self.fixture.workspace / 'copy')
        self.assertFalse((self.fixture.workspace / 'copy').exists())

    def test_missing_requested_order_falls_back_but_other_order_reuses(self):
        self.config['experiments']['configuration']['orders'].remove('Gate-Screen-Audit')
        self.prepare()
        self.runner.run(6)
        root = self.fixture.results / 'threshold/seed_42/recall_0.999'
        self.assertEqual(read_json(root / 'Screen-Gate-Audit/execution.json')['result_source'], 'inherited_item_2')
        other = read_json(root / 'Gate-Screen-Audit/execution.json')
        self.assertEqual(other['result_source'], 'new_item_6_inference')
        self.assertIn('Missing Experiment 2 configuration', other['reuse_fallback_reason'])


if __name__ == '__main__':
    unittest.main()
