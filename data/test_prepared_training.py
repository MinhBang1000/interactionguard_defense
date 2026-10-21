"""Fixed training artifact tests; no model imports or training required."""
import json
from pathlib import Path
import tempfile
import unittest

from data.data_preprocess import ROOT, expand_trace, read_jsonl, sha256, write_jsonl
from data.prepare_training import load_prepared, prepare
from data.test_preprocessing import episode
from data.training_policy import select_training_rows


class PreparedTrainingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.processed = self.root / "processed"
        self.output = self.root / "training"
        for split in ("train", "val"):
            rows = []
            for kind in ("tool", "correlated"):
                trace = episode(kind)
                trace["id"] = f"{split}_{kind}"
                rows.extend(expand_trace(trace))
            write_jsonl(self.processed / f"{split}_processed.jsonl", rows)

    def test_materialized_rows_equal_policy_and_load_verbatim(self):
        manifest = prepare(self.processed, self.output)
        for split in ("train", "val"):
            source = read_jsonl(self.processed / f"{split}_processed.jsonl")
            for stage in ("encoder", "autoencoder"):
                rows = load_prepared(stage, split, output_dir=self.output)
                self.assertEqual(rows, select_training_rows(source, benign_only=stage == "autoencoder"))
                self.assertFalse(any(r["attack_type"] == "correlated" for r in rows))
                self.assertFalse(any(r["annotation_requires_review"] for r in rows))
                if stage == "autoencoder":
                    self.assertTrue(all(r["label"] == 0 for r in rows))
                self.assertEqual(manifest["artifacts"][f"{stage}/{split}.jsonl"]["rows"], len(rows))

    def test_preparation_is_deterministic(self):
        first = prepare(self.processed, self.output)
        second = prepare(self.processed, self.output)
        self.assertEqual(first, second)

    def test_loader_rejects_modified_artifact_without_filtering(self):
        prepare(self.processed, self.output)
        path = self.output / "autoencoder/train.jsonl"
        rows = read_jsonl(path)
        rows[0]["attack_type"] = "correlated"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "modified"):
            load_prepared("autoencoder", "train", output_dir=self.output)

    def test_loader_rejects_changed_source(self):
        prepare(self.processed, self.output)
        path = self.processed / "train_processed.jsonl"
        write_jsonl(path, read_jsonl(path)[:-1])
        with self.assertRaisesRegex(ValueError, "source changed"):
            load_prepared("encoder", "train", output_dir=self.output)

    def test_missing_manifest_does_not_fallback_to_processed(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            load_prepared("encoder", "train", output_dir=self.output)

    def test_train_val_episode_overlap_is_rejected(self):
        write_jsonl(self.processed / "val_processed.jsonl", read_jsonl(self.processed / "train_processed.jsonl"))
        with self.assertRaisesRegex(ValueError, "episode IDs overlap"):
            prepare(self.processed, self.output)

    def test_path_override_cannot_bypass_preparation(self):
        prepare(self.processed, self.output)
        with self.assertRaisesRegex(ValueError, "must match"):
            load_prepared("encoder", "train", output_dir=self.output,
                          expected_path=self.processed / "train_processed.jsonl")


@unittest.skipUnless((ROOT / "data/training/manifest.json").exists(), "Run data preparation first")
class GeneratedTrainingTests(unittest.TestCase):
    def test_every_export_matches_processed_selection_and_hash(self):
        manifest = json.loads((ROOT / "data/training/manifest.json").read_text())
        for split in ("train", "val"):
            rows = read_jsonl(ROOT / f"data/processed/{split}_processed.jsonl")
            for stage in ("encoder", "autoencoder"):
                actual = load_prepared(stage, split)
                expected = select_training_rows(rows, benign_only=stage == "autoencoder")
                self.assertEqual(actual, expected)
                path = ROOT / "data/training" / stage / f"{split}.jsonl"
                self.assertEqual(sha256(path), manifest["artifacts"][f"{stage}/{split}.jsonl"]["sha256"])


if __name__ == "__main__":
    unittest.main()
