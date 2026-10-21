"""Episode ASR regressions; synthetic decisions only, never external API calls."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from data.data_preprocess import expand_trace, write_json, write_jsonl, sha256
from data.test_preprocessing import episode
from eval.asr import score_episode, summarize, load_decisions, episode_hash
from eval.prompt_success_judge import judge_episode


class ASRTests(unittest.TestCase):
    def setUp(self):
        self.trace = episode()

    def full(self, block=None):
        return [(i, int(i == block)) for i in range(6)]

    def test_no_defense_success(self):
        self.assertTrue(score_episode(self.trace)["surviving_success"])

    def test_timely_block(self):
        result = score_episode(self.trace, self.full(3))
        self.assertFalse(result["surviving_success"])
        self.assertTrue(result["timely_prevention"])

    def test_block_at_assistant_commit_is_not_credited(self):
        self.assertTrue(score_episode(self.trace, self.full(4))["surviving_success"])

    def test_block_at_result_is_too_late(self):
        self.assertTrue(score_episode(self.trace, self.full(5))["surviving_success"])

    def test_untimed_block_is_unknown(self):
        self.assertIsNone(score_episode(self.trace, [(None, 1)])["surviving_success"])

    def test_missing_predictions_are_unknown(self):
        self.assertIsNone(score_episode(self.trace, [])["surviving_success"])

    def test_partial_timely_block_suffices(self):
        self.assertFalse(score_episode(self.trace, [(1, 1)])["surviving_success"])

    def test_partial_late_block_does_not_prove_no_earlier_block(self):
        self.assertIsNone(score_episode(self.trace, [(5, 1)])["surviving_success"])

    def test_natural_failure_not_prevention(self):
        trace = copy.deepcopy(self.trace)
        trace["reasoning_steps"] = trace["reasoning_steps"][:3]
        trace["trace"] = trace["trace"][:3]
        result = score_episode(trace, [(1, 1)])
        self.assertFalse(result["surviving_success"])
        self.assertFalse(result["timely_prevention"])

    def test_pre_injection_block_is_separate(self):
        result = score_episode(self.trace, self.full(0))
        self.assertTrue(result["pre_injection_block"])
        self.assertTrue(result["timely_prevention"])

    def test_prompt_without_judge_is_unknown(self):
        self.trace["attack_type"] = "prompt"
        self.assertIsNone(score_episode(self.trace)["baseline_success"])

    def test_judge_is_trace_bound(self):
        self.trace["attack_type"] = "prompt"
        judge = {"sample": {"episode_sha256": "stale", "verdict": "SUCCESS"}}
        self.assertIsNone(score_episode(self.trace, judges=judge)["baseline_success"])

    def test_prompt_success_timing(self):
        self.trace["attack_type"] = "prompt"
        judge = {"sample": {"episode_sha256": episode_hash(self.trace), "verdict": "SUCCESS", "first_success_step": 2}}
        self.assertFalse(score_episode(self.trace, self.full(1), judge)["surviving_success"])
        self.assertTrue(score_episode(self.trace, self.full(2), judge)["surviving_success"])

    def test_unknown_timing_not_treated_as_prevention(self):
        self.trace["attack_type"] = "prompt"
        judge = {"sample": {"episode_sha256": episode_hash(self.trace), "verdict": "SUCCESS"}}
        self.assertIsNone(score_episode(self.trace, self.full(1), judge)["surviving_success"])
        self.assertTrue(score_episode(self.trace, self.full(), judge)["surviving_success"])

    def test_all_attempt_denominator_and_bounds(self):
        a = score_episode(self.trace)
        b = copy.deepcopy(a)
        b.update(id="failed", baseline_success=False, surviving_success=False)
        c = copy.deepcopy(a)
        c.update(id="unknown", baseline_success=None, surviving_success=None)
        stats = summarize([a, b, c])["global"]
        self.assertEqual(stats["total"], 3)
        self.assertIsNone(stats["asr"])
        self.assertEqual(stats["asr_lower_bound"], 1/3)
        self.assertEqual(stats["asr_upper_bound"], 2/3)

    def test_benign_episode_false_blocking(self):
        self.trace["attack_type"] = None
        report = summarize([score_episode(self.trace, self.full(0))])
        self.assertEqual(report["benign"]["false_block_rate"], 1)
        self.assertEqual(report["global"]["total"], 0)

    def test_trace_result_disagreement_is_unknown(self):
        self.trace["trace"] = []
        self.assertIsNone(score_episode(self.trace)["baseline_success"])

    def test_duplicate_or_invalid_verdict_rejected(self):
        for decisions in [[(1, 0), (1, 1)], [(1, 2)], [(9, 0)], [(None, 0), (1, 1)]]:
            with self.assertRaises(ValueError):
                score_episode(self.trace, decisions)

    def test_prediction_provenance_and_id_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            processed = Path(directory) / "processed.jsonl"
            output = Path(directory) / "pred.json"
            rows = list(expand_trace(self.trace))
            write_jsonl(processed, rows)
            write_json(output, {"predictions": [1, 0], "sample_ids": [rows[3]["id"], rows[0]["id"]], "processed_sha256": sha256(processed)})
            self.assertEqual(load_decisions(output, processed)["sample"], [(3, 1), (0, 0)])
            write_json(output, {"predictions": [0]*6})
            with self.assertRaises(ValueError):
                load_decisions(output, processed)

    def test_judge_failure_never_invented_on_api_error(self):
        class Broken:
            def create(self, **kwargs):
                raise RuntimeError("offline")
        client = SimpleNamespace(chat=SimpleNamespace(completions=Broken()))
        self.assertEqual(judge_episode(self.trace, client)["verdict"], "UNKNOWN")

    def test_judge_rejects_fabricated_success_evidence(self):
        result = json.dumps({"verdict": "SUCCESS", "evidence": "invented secret", "first_success_step": 2})
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=result))]))))
        self.assertEqual(judge_episode(self.trace, client)["verdict"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
