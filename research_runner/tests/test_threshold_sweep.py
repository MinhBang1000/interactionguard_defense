"""CPU-only checks for the Item 6 operating-point policy."""
from copy import deepcopy
import itertools
from pathlib import Path
import tempfile
import unittest

import yaml

from research_runner.config import ConfigError, load_config
from research_runner.scoring import calibrate_gate
from research_runner.tests.test_scoring import reference_metrics
from research_runner.threshold_sweep import POLICY, select_gate_threshold, validate_settings


class SweepSelectionTests(unittest.TestCase):
    def test_highest_empirical_threshold_matches_exhaustive_search_with_ties(self):
        for labels in itertools.product((0, 1), repeat=4):
            if set(labels) != {0, 1}:
                continue
            for scores in itertools.product((-1., 0., 1.), repeat=4):
                cutoffs = [min(scores) - max(1e-12, abs(min(scores)) * 1e-6), *sorted(set(scores))]
                for floor in (.25, .5, .9, .999, 1.):
                    feasible = [(t, reference_metrics(labels, scores, t)) for t in cutoffs
                                if reference_metrics(labels, scores, t)['recall'] >= floor]
                    expected, metrics = max(feasible)
                    result = select_gate_threshold(labels, scores, floor)
                    self.assertEqual(result['threshold'], expected)
                    self.assertEqual(result['metrics'], metrics)
                    self.assertEqual(result['objective'], POLICY)
                    self.assertEqual(result['comparison'], 'strict_greater_than')
                    self.assertEqual(result['validation_escalation_ratio'], (metrics['tp'] + metrics['fp']) / 4)

    def test_sweep_does_not_collapse_to_constrained_max_f1_and_leaves_calibration_unchanged(self):
        labels, scores = [1, 0, 1, 1, 1], [.1, .2, .3, .4, .5]
        original = calibrate_gate(labels, scores, .5)
        sweep = select_gate_threshold(labels, scores, .5)
        self.assertEqual(original['metrics']['recall'], 1.)
        self.assertEqual(sweep['threshold'], .3)
        self.assertEqual(sweep['metrics']['recall'], .5)
        self.assertEqual(sweep['metrics']['fpr'], 0.)
        self.assertEqual(calibrate_gate(labels, scores, .5), original)

    def test_strict_boundary_and_equal_scores_are_not_split(self):
        result = select_gate_threshold([0, 1], [.5, .5], .999)
        self.assertLess(result['threshold'], .5)
        self.assertEqual(result['metrics']['recall'], 1.)
        self.assertEqual(result['metrics']['fpr'], 1.)

    def test_invalid_inputs_and_floor_rejected(self):
        for floor in (0, -1, 1.1, float('nan'), True, '0.9'):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                select_gate_threshold([0, 1], [.1, .9], floor)
        for labels, scores in (([], []), ([0], [.1]), ([0, 1], [.1]), ([0, 1], [float('nan'), .9])):
            with self.assertRaises(ValueError):
                select_gate_threshold(labels, scores, .9)

    def test_checked_in_grid_policy_and_main_calibration(self):
        config = load_config(Path(__file__).resolve().parents[2] / 'config/experiments.yaml')
        settings = config['experiments']['threshold']
        self.assertEqual(settings['recall_floors'], [.9, .95, .98, .999])
        self.assertEqual(settings['selection_policy'], POLICY)
        self.assertEqual(settings['reuse_configuration_recall_floors'], [.999])
        self.assertTrue(settings['include_validation_max_f1'])
        self.assertEqual(settings['orders'], ['Screen-Gate-Audit', 'Gate-Screen-Audit'])
        self.assertEqual(config['training']['calibration']['gate_recall_floor'], .999)
        for change in ({'selection_policy': 'test_max_f1'}, {'recall_floors': [.9, .9]},
                       {'recall_floors': [.99, .9]}, {'recall_floors': [0, .9]}):
            value = deepcopy(config)
            value['experiments']['threshold'].update(change)
            for key in tuple(value):
                if key.startswith('_'):
                    value.pop(key)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / 'config.yaml'
                path.write_text(yaml.safe_dump(value))
                with self.subTest(change=change), self.assertRaises(ConfigError):
                    load_config(path)

    def test_invalid_policy_and_duplicate_targets_rejected(self):
        for setting in ({'selection_policy': 'min_test_asr'}, {'recall_floors': [.9, .9]},
                        {'recall_floors': [.99, .9]}, {'recall_floors': [0]}, {'recall_floors': []},
                        {'recall_floors': [.9], 'reuse_configuration_recall_floors': [.999]},
                        {'recall_floors': [.9], 'reuse_configuration_recall_floors': True}):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                validate_settings(setting)


if __name__ == '__main__':
    unittest.main()
