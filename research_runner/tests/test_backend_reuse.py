"""Real orchestration/report serialization with a frozen Item 2 reference."""
from copy import deepcopy
from pathlib import Path
import unittest

from research_runner.artifacts import read_json, read_jsonl, write_jsonl
from research_runner.backend_reuse import REPORT_FILES, load_reference
from research_runner.tests import test_engine as fixtures


class BackendReferenceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.EngineTests(methodName='runTest')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.addCleanup(self.fixture.tearDown)
        self.runner = self.fixture.runner
        self.config = self.fixture.config

    def test_zero_llama_calls_and_identical_reports_for_each_seed(self):
        self.fixture.prepare(full=True)
        before = self.fixture.snapshot_artifacts()
        calls = len(self.fixture.backend.calls)
        self.runner.run(5)
        new = self.fixture.backend.calls[calls:]
        self.assertEqual(len(new), 3 * len(self.config['training']['seeds']))
        self.assertTrue(all(call[0] == 'infer_audit' for call in new))
        self.assertFalse(any(call[1]['model'] == self.config['models']['audit']['model'] for call in new))
        output = self.fixture.results / 'backend'
        protocol = read_json(output / 'comparison_protocol.json')
        self.assertEqual(protocol['llama31_8b']['new_inference_trials'], 0)
        summary = read_json(output / 'summary.json')
        self.assertEqual(summary['llama31_8b']['new_inference_trials'], 0)
        self.assertIn('inherited_item_2', summary['llama31_8b']['variation'])
        for seed in self.config['training']['seeds']:
            original = self.fixture.results / f'configuration_stage2/seed_{seed}/Screen-Gate-Audit'
            reused = output / f'seed_{seed}/llama31_8b'
            for filename in REPORT_FILES:
                self.assertEqual((original / filename).read_bytes(), (reused / filename).read_bytes())
            provenance = read_json(reused / 'reuse.json')
            self.assertEqual(provenance['new_inference_requests'], 0)
            self.assertTrue(provenance['latency_inherited_from_item_2'])
            self.assertFalse(provenance['new_latency_measurement'])
            self.assertEqual(read_json(reused / 'audit/timing.json')['submitted_requests'], 0)
        after = self.fixture.snapshot_artifacts()
        self.assertTrue(all(after[path] == digest for path, digest in before.items()))
        self.fixture.verify_backends(self.fixture.results / 'configuration_stage1')

    def test_missing_stage2_stops_without_fallback_inference(self):
        self.fixture.prepare(training=True)
        self.runner.configuration_stage1()
        before = len(self.fixture.backend.calls)
        with self.assertRaisesRegex(ValueError, 'configuration_stage2'):
            self.runner.run(5)
        self.assertEqual(before, len(self.fixture.backend.calls))

    def test_fresh_primary_inference_is_forbidden(self):
        self.fixture.prepare(full=True)
        self.config['experiments']['backend']['candidates'][0].pop('reuse_configuration')
        before = len(self.fixture.backend.calls)
        with self.assertRaisesRegex(ValueError, 'new inference is forbidden'):
            self.runner.run(5)
        self.assertEqual(before, len(self.fixture.backend.calls))

    def test_reference_candidate_cannot_override_model_or_threshold(self):
        self.fixture.prepare(full=True)
        original = deepcopy(self.config['experiments']['backend']['candidates'][0])
        before = len(self.fixture.backend.calls)
        for changes in ({'threshold': .9}, {'model': 'different/checkpoint'}, {'vllm_version': '0.32.0'}):
            self.config['experiments']['backend']['candidates'][0] = {**original, **changes}
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.runner.run(5)
        self.assertEqual(before, len(self.fixture.backend.calls))

    def test_reference_wrong_payload_or_order_is_rejected(self):
        self.fixture.prepare(full=True)
        first, full = (self.fixture.results / name for name in ('configuration_stage1', 'configuration_stage2'))
        rows = self.runner.rows()
        staged, requests, _, _ = self.runner._backend_inputs(first, 42, rows, 'Screen-Gate-Audit')
        source = full / 'seed_42/audit/aligned_responses.jsonl'
        original = read_jsonl(source)
        index = next(i for i, row in enumerate(original) if row['configuration'] == 'Screen-Gate-Audit')
        for field, value in (('payload', 'wrong payload'), ('seed', 999), ('request_id', 'wrong hash')):
            altered = deepcopy(original)
            altered[index][field] = value
            write_jsonl(source, altered)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'mismatch'):
                load_reference(self.runner, full, first, 42, rows, staged, requests)
        altered = original + [original[index]]
        write_jsonl(source, altered)
        with self.assertRaisesRegex(ValueError, 'missing/duplicate/misordered'):
            load_reference(self.runner, full, first, 42, rows, staged, requests)


if __name__ == '__main__':
    unittest.main()
