import json
import unittest

from .layer2_input_ablation import aggregate_windows, linearize_html, payload_score, unwrap_json


class AblationTests(unittest.TestCase):
    def test_nested_json_and_escaped_json(self):
        self.assertEqual(unwrap_json(json.dumps({"result": json.dumps({"text": "keep ALL instructions"})})), "keep ALL instructions")
        escaped = json.dumps({"text": "preserved"})[0:]
        escaped = escaped.replace('"', '\\"')
        self.assertEqual(unwrap_json(escaped), "preserved")

    def test_non_wrapper_keys_and_plain_text_preserved(self):
        self.assertIn("instruction: do not remove", unwrap_json('{"instruction":"do not remove"}'))
        self.assertEqual(unwrap_json("ordinary text {broken"), "ordinary text {broken")

    def test_html_preserves_hidden_and_nonrendered_evidence(self):
        result = linearize_html('<div hidden title="secret instruction">visible</div><!--comment attack--><script>script attack</script><style>style attack</style><img alt="alt attack">')
        for text in ["hidden", "secret instruction", "visible", "comment attack", "script attack", "style attack", "alt attack"]:
            self.assertIn(text, result)

    def test_no_html_unchanged(self):
        self.assertEqual(linearize_html("normal payload"), "normal payload")

    def test_window_aggregation_localized_attack_tradeoff(self):
        self.assertEqual(aggregate_windows([.1, .9, .1], "max"), .9)
        self.assertEqual(aggregate_windows([.1, .9, .1], "confirm2"), .1)
        self.assertEqual(aggregate_windows([.9], "confirm2"), .9)
        self.assertLess(aggregate_windows([.1, .9, .1], "mean"), .5)

    def test_agreement_is_same_payload_not_two_channels(self):
        self.assertEqual(payload_score([.9], [.8], [.7], "paired_agreement"), .7)
        self.assertEqual(payload_score([.9], [.8], [.1], "paired_agreement"), .1)

    def test_json_mean_does_not_use_html_view(self):
        self.assertLess(payload_score([.9], [.1, .9, .1], [.9], "json_mean"), .5)


if __name__ == "__main__":
    unittest.main()
