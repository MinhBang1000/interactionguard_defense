"""Optional read-only replay of archived verdicts, never fresh model evaluation."""
import hashlib
from pathlib import Path
import unittest

from eval.asr import load_judges
from research_runner.artifacts import read_json, read_jsonl, file_sha256
from research_runner.scoring import route, metrics, score_episodes

ROOT = Path(__file__).resolve().parents[2]


class ReferenceReplayTests(unittest.TestCase):
    def test_all_five_reference_gates_and_strict_asr_are_reproduced_without_models(self):
        reference = ROOT / 'results/side_retrained_l1_deepset_l3/20261021'
        required = [reference / 'config.json', reference / 'inputs.jsonl', reference / 'l3_outputs.jsonl',
                    ROOT / 'data/processed/test_processed.jsonl', ROOT / 'data/splits/test.jsonl']
        if not all(path.is_file() for path in required):
            self.skipTest('Optional local side-test/data artifacts are not distributed in Git')
        config = read_json(reference / 'config.json')
        self.assertEqual(file_sha256(required[-2]), config['test_sha256'])
        rows = read_jsonl(required[-2])
        inputs = read_jsonl(reference / 'inputs.jsonl')
        l1 = {r['id']: {'score': r['l1_error']} for r in inputs}
        l2 = {r['id']: {'score': r['l2_probability']} for r in inputs}
        cached = {r['payload_sha256']: r for r in read_jsonl(reference / 'l3_outputs.jsonl')}
        l3 = {r['id']: cached[hashlib.sha256(r['payload'].encode()).hexdigest()]
              for r in inputs if hashlib.sha256(r['payload'].encode()).hexdigest() in cached}
        self.assertEqual(set(l1), {r['id'] for r in rows})
        strict = None
        for path in sorted(reference.glob('threshold_*.json')):
            expected = read_json(path)
            if 'full_cascade' not in expected:
                continue
            decisions = route(rows, l1, l2, ['L1', 'L2', 'L3'], expected['l1_threshold'], config['l2_threshold'], l3)
            actual = metrics(decisions)
            for key, value in expected['full_cascade'].items():
                self.assertAlmostEqual(actual[key], value, places=12, msg=f'{path.name}/{key}')
            if expected['l1_threshold'] == min(config['thresholds']):
                strict = decisions
        self.assertIsNotNone(strict)
        report = score_episodes(read_jsonl(required[-1]), strict,
                                load_judges(ROOT / 'eval/prompt_success_judgements.jsonl'))
        self.assertEqual(report['global']['total'], 666)
        self.assertEqual(report['global']['success'], 25)
        self.assertEqual({k: r['success'] for k, r in report['per_attack_type'].items()},
                         {'prompt': 0, 'rag': 23, 'tool': 2, 'correlated': 0})
        self.assertEqual(report['benign']['blocked'], 8)


if __name__ == '__main__':
    unittest.main()
