import json
from pathlib import Path
import unittest

from .prepare_fewshot import normalized


class FewshotTests(unittest.TestCase):
    def test_normalized_comparison(self):
        self.assertEqual(normalized("  HELLO\nworld "), "hello world")

    def test_fixed_examples_are_balanced_and_consistent(self):
        examples = json.loads(Path(__file__).with_name("synthetic_examples.json").read_text())
        self.assertEqual(len(examples), 8)
        self.assertEqual(sum(x["output"]["is_malicious"] for x in examples), 4)
        for example in examples:
            v = example["output"]
            self.assertEqual(v["is_malicious"], any(v[k] for k in ["prompt_attack", "retrieve_attack", "tool_attack", "agent_adoption"]))
            self.assertTrue(.65 <= v["confidence"] <= 1 if v["is_malicious"] else 0 <= v["confidence"] <= .35)
            self.assertLess(len(v["user_intent"].split()), 12)
            self.assertLess(len(v["reason"].split()), 18)
            self.assertLess(len(v["evidence_span"].split()), 20)
            if v["is_malicious"]:
                self.assertTrue(v["evidence_span"])
                self.assertIn(v["evidence_span"], example["input"])
            else:
                self.assertEqual(v["evidence_span"], "")


if __name__ == "__main__":
    unittest.main()
