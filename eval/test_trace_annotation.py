import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from data.data_preprocess import read_jsonl, write_jsonl, normalize_trace
from data.test_preprocessing import episode
from eval.annotate_trace_outcomes import annotation, append_only, preserve_existing, main, FIELD
from eval.asr import episode_hash, original_outcome


class AnnotationTests(unittest.TestCase):
    def test_annotations_do_not_mutate_original(self):
        trace = episode()
        before = copy.deepcopy(trace)
        value = annotation(trace, "correlated", {}, None)
        self.assertEqual(trace, before)
        self.assertEqual(value["verdict"], "SUCCESS")

    def test_benign_is_not_applicable(self):
        self.assertEqual(annotation(episode(), None, {}, None)["verdict"], "NOT_APPLICABLE")

    def test_missing_prompt_judge_is_unknown(self):
        self.assertEqual(annotation(episode("prompt"), "prompt", {}, None)["verdict"], "UNKNOWN")

    def test_frozen_outcome_is_used_and_idempotent(self):
        trace = episode()
        value = annotation(trace, "correlated", {}, None)
        trace[FIELD] = value
        self.assertEqual(annotation(trace, "correlated", {}, None), value)
        self.assertTrue(original_outcome(trace)["success"])

    def test_changed_trace_or_contract_invalidates_frozen_outcome(self):
        for edit in ["trace", "contract"]:
            trace = episode()
            trace[FIELD] = annotation(trace, "correlated", {}, None)
            if edit == "trace":
                trace["prompt"] = "changed"
            else:
                trace[FIELD]["contract"] = "stale"
            with self.assertRaises(ValueError):
                original_outcome(trace)

    def test_reannotation_preserves_source_provenance_but_rejects_outcome_changes(self):
        value = annotation(episode(), "correlated", {}, None)
        edited = {**value, "assessment_sha256": "new_nonsemantic_source_revision"}
        self.assertEqual(preserve_existing(edited, value), value)
        with self.assertRaises(ValueError):
            preserve_existing({**edited, "verdict": "UNKNOWN", "attack_success": None}, value)

    def test_only_added_field_is_written(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "trace.jsonl"
            trace = episode()
            write_jsonl(path, [trace])
            added = {**trace, FIELD: {"verdict": "UNKNOWN"}}
            append_only(path, [trace], [added])
            self.assertEqual(read_jsonl(path), [added])
            bad = {**added, "prompt": "changed"}
            with self.assertRaises(ValueError):
                append_only(path, [added], [bad])

    def test_split_regeneration_restores_frozen_outcomes_from_backup(self):
        from data.data_split import SOURCES
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name in ["eval/asr.py", "data/data_preprocess.py"]:
                path = root/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture policy source\n")
            traces = []
            raw_bytes = {}
            for filename, kind in SOURCES.items():
                trace = episode(kind)
                trace["id"] = kind or "benign"
                trace["label"] = int(kind is not None)
                path = root/"data/raw"/filename
                write_jsonl(path, [trace])
                raw_bytes[path] = path.read_bytes()
                traces.append(normalize_trace(trace))
            for split, rows in [("train", traces), ("val", []), ("test", [])]:
                write_jsonl(root/f"data/splits/{split}.jsonl", rows)
            args = ["--apply", "--judges", str(root/"missing_judges.jsonl")]
            with patch("eval.annotate_trace_outcomes.ROOT", root), patch("eval.asr.ROOT", root), redirect_stdout(io.StringIO()):
                main(args)
                frozen = read_jsonl(root/"data/splits/train.jsonl")
                backup = root/"frozen.zip"
                with zipfile.ZipFile(backup, "w") as archive:
                    for split in ["train", "val", "test"]:
                        path = root/f"data/splits/{split}.jsonl"
                        archive.write(path, f"data/splits/{split}.jsonl")
                write_jsonl(root/"data/splits/train.jsonl", traces)
                main(args + ["--frozen-backup", str(backup)])
                self.assertEqual(read_jsonl(root/"data/splits/train.jsonl"), frozen)
            for path, before in raw_bytes.items():
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
