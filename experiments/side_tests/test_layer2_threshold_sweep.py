import unittest

import numpy as np

from .layer2_threshold_sweep import evaluate_point, select_fpr


class ThresholdTests(unittest.TestCase):
    def test_ties_not_split_and_strict_threshold(self):
        result = select_fpr([0, 0, 1, 1], [.9, .9, .9, .99], 0)
        self.assertEqual(result["threshold"], .9)
        self.assertEqual(result["validation"]["tp"], 1)
        self.assertEqual(result["validation"]["fp"], 0)

    def test_nonempty_unavailable(self):
        result = select_fpr([0, 1], [1., .9], 0)
        self.assertFalse(result["nonempty_true_positive_coverage"])
        self.assertEqual(result["threshold"], 1.)

    def test_exact_budget(self):
        result = select_fpr([0, 0, 1, 1], [.1, .8, .9, .7], .5)
        self.assertEqual(result["validation"]["tp"], 2)
        self.assertEqual(result["validation"]["fp"], 1)

    def test_conditional_and_global_fpr_differ(self):
        result = evaluate_point(np.array([1, 1, 0, 0]), [.1, .9, .9, .1], [.99, .9, .9, .99], .5, .8)
        self.assertEqual(result["fp"], 1)
        self.assertEqual(result["fpr"], .5)
        self.assertEqual(result["conditional_l2_fpr"], 1.)
        self.assertEqual(result["l1_bypassed_positive"], 1)

    def test_pending_not_blocked_and_empty_precision(self):
        result = evaluate_point([1, 0], [1., 1.], [.9, .9], .5, .95)
        self.assertEqual(result["l3_pending"], 2)
        self.assertFalse(result["precision_defined"])

    def test_reject_invalid(self):
        with self.assertRaises(ValueError):
            select_fpr([0, 1], [.9, float("nan")], .01)


if __name__ == "__main__":
    unittest.main()
