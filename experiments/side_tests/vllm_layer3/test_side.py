import unittest

from experiments.side_tests.vllm_layer3.run import binary_metrics


class SideMetricsTests(unittest.TestCase):
    def test_confusion(self):
        result = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertEqual([result[k] for k in ("tp", "fp", "fn", "tn")], [1, 1, 1, 1])
        self.assertEqual(result["f1"], .5)

    def test_empty(self):
        self.assertEqual(binary_metrics([], [])["f1"], 0)

    def test_clean_prompt_preserves_policy_not_examples(self):
        from experiments.side_tests.vllm_layer3.prepare import clean_prompt
        from models.detectors.layer3_detector_o2 import Layer3DetectorO2
        original = Layer3DetectorO2.__new__(Layer3DetectorO2)._build_system_prompt()
        cleaned = clean_prompt(original)
        for text in ("Strong malicious examples", "get_information", '"is_malicious": true'):
            self.assertNotIn(text, cleaned)
        for text in ("prompt_attack", "retrieve_attack", "tool_attack", "agent_adoption", "confidence"):
            self.assertIn(text, cleaned)


if __name__ == "__main__":
    unittest.main()
