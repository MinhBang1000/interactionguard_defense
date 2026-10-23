import unittest

from .layer2_full_test_sweep import MODELS, THRESHOLDS, summarize, verify_cached_inputs


class FullSweepTests(unittest.TestCase):
    def test_registry_and_grid(self):
        self.assertEqual(len(MODELS), 6)
        self.assertEqual(len({m[0] for m in MODELS}), 6)
        self.assertEqual(THRESHOLDS, sorted(set(THRESHOLDS)))
        self.assertEqual(len(THRESHOLDS), 24)

    def test_strict_threshold_and_no_positive_prediction(self):
        result = summarize([0, 1], [.5, .5], .5)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (0, 0, 1, 1))
        self.assertFalse(result["precision_defined"])
        self.assertEqual(result["fnr"], 1.)

    def test_missing_positive_class_not_perfect_recall(self):
        result = summarize([0, 0], [.1, .9], .5)
        self.assertFalse(result["recall_defined"])
        self.assertIsNone(result["fnr"])

    def test_cache_allows_metadata_cleanup_only(self):
        old = [{"id": "x", "sequence": "same text", "label": 1, "episode_attack_success": True}]
        current = [{"id": "x", "sequence": "same text", "label": 1}]
        verify_cached_inputs(old, current)
        for field, value in [("id", "y"), ("sequence", "changed"), ("label", 0)]:
            with self.assertRaises(ValueError):
                verify_cached_inputs(old, [{**current[0], field: value}])


if __name__ == "__main__":
    unittest.main()
