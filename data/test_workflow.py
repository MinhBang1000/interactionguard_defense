"""Verify menu workflow stage order without touching project datasets."""
from contextlib import ExitStack, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data.workflow import main


class WorkflowTests(unittest.TestCase):
    def run_scope(self, scope):
        calls = []
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            root = Path(directory)
            stack.enter_context(patch("data.workflow.backup_current", side_effect=lambda: calls.append("backup") or root/"backup.zip"))
            for module, name in [("data.data_split", "split"), ("data.data_preprocess", "prefix"),
                                 ("data.prepare_training", "training"), ("data.validate_preprocessing", "validate")]:
                stack.enter_context(patch(module+".main", side_effect=lambda *args, n=name, **kwargs: calls.append(n)))
            def annotate(args):
                calls.append("annotate")
                return {"run": str(root/"outcomes")}
            stack.enter_context(patch("data.annotate_raw.main", side_effect=annotate))
            stack.enter_context(patch("data.workflow.subprocess.run", side_effect=lambda *args, **kwargs: calls.append("tests")))
            stack.enter_context(patch("eval.asr.evaluate_files", side_effect=lambda *args, **kwargs: calls.append("asr_report") or {"per_attack_type": {}}))
            with redirect_stdout(io.StringIO()):
                main(["--scope", scope])
        return calls

    def test_full_stage_order(self):
        self.assertEqual(self.run_scope("all"),
                         ["backup", "annotate", "split", "prefix", "training", "tests", "validate", "asr_report"])

    def test_asr_only_does_not_regenerate_prefix_or_training(self):
        self.assertEqual(self.run_scope("asr"), ["backup", "annotate"])

    def test_split_is_an_isolated_step(self):
        self.assertEqual(self.run_scope("split"), ["backup", "split"])

    def test_preprocess_is_an_isolated_step(self):
        self.assertEqual(self.run_scope("preprocess"), ["backup", "prefix"])

    def test_training_only_does_not_change_traces(self):
        self.assertEqual(self.run_scope("training"), ["backup", "training"])

    def test_validation_is_an_isolated_step(self):
        self.assertEqual(self.run_scope("validate"), ["backup", "tests", "validate"])


if __name__ == "__main__":
    unittest.main()
