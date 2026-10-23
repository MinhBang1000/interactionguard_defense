import unittest

from .detector_survey import malicious_index


class MappingTests(unittest.TestCase):
    def test_explicit_mappings(self):
        self.assertEqual(malicious_index({0: "benign", 1: "jailbreak"}, "sentinel")[0], 1)
        self.assertEqual(malicious_index({0: "INJECTION", 1: "SAFE"}, "other")[0], 0)

    def test_prompt_guard_official_override(self):
        index, source = malicious_index({0: "LABEL_0", 1: "LABEL_1"}, "prompt_guard_2")
        self.assertEqual(index, 1)
        self.assertIn("meta-llama/llama-cookbook", source)

    def test_no_guessing_for_other_models(self):
        with self.assertRaises(ValueError):
            malicious_index({0: "LABEL_0", 1: "LABEL_1"}, "unknown")


if __name__ == "__main__":
    unittest.main()
