"""Calibration, deployment safety, import safety and accumulation regressions."""
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from data.data_preprocess import sha256, write_json
from training.layer1_common import calibrate, directory_hash, metrics


class CalibrationTests(unittest.TestCase):
    def test_separable_scores(self):
        result = calibrate([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result["metrics"]["f1"], 1)
        self.assertEqual(result["metrics"]["recall"], 1)

    def test_ties_not_split(self):
        result = calibrate([0, 1], [0.5, 0.5], 1)
        self.assertEqual(result["metrics"]["recall"], 1)
        self.assertEqual(result["metrics"]["fpr"], 1)

    def test_float32_all_positive_threshold_representable(self):
        errors = np.array([0.001, 0.001], dtype=np.float32)
        result = calibrate([0, 1], errors, 1)
        self.assertTrue((errors > result["threshold"]).all())

    def test_recall_constraint_changes_operating_point(self):
        result = calibrate([1, 0, 0, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9], 1)
        self.assertEqual(result["metrics"]["recall"], 1)
        self.assertGreater(result["unconstrained"]["metrics"]["f1"], result["metrics"]["f1"])

    def test_calibration_agrees_with_exhaustive_metrics(self):
        rng = np.random.default_rng(42)
        errors = rng.integers(0, 10, 100).astype(float)
        labels = rng.integers(0, 2, 100)
        result = calibrate(labels, errors, 0.9)
        choices = [metrics(labels, errors, threshold) for threshold in [-1]+list(np.unique(errors))]
        expected = max(m["f1"] for m in choices if m["recall"] >= 0.9)
        self.assertEqual(result["metrics"]["f1"], expected)

    def test_invalid_calibration_data(self):
        for labels, scores in [([], []), ([0], [1]), ([0, 1], [np.nan, 1]), ([0, 1], [1])]:
            with self.assertRaises(ValueError):
                calibrate(labels, scores)

    def test_encoder_fingerprint_changes_with_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            path.write_bytes(b"first")
            first = directory_hash(directory)
            path.write_bytes(b"second")
            self.assertNotEqual(first, directory_hash(directory))

    def test_imports_do_not_start_training(self):
        import training.autoencoder_training as ae
        import training.modernbert_finetune as encoder
        self.assertTrue(callable(ae.run))
        self.assertFalse(hasattr(ae, "train_emb"))
        model = encoder.configure_loss_normalization(SimpleNamespace())
        self.assertFalse(model.accepts_loss_kwargs)

    def test_gpu_preflight_refuses_busy_device(self):
        from training.layer1_pipeline import gpu_preflight
        with patch("subprocess.check_output", return_value="0, GPU-id, NVIDIA GeForce RTX 4090, 24564, 22400"):
            with self.assertRaises(RuntimeError):
                gpu_preflight("0")

    def test_promotion_preserves_other_layers_and_has_backup(self):
        from training.layer1_pipeline import promote
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config/detection_config.yaml"
            config.parent.mkdir()
            config.write_text('layers:\n  layer1:\n    params:\n      bert_model_path: old\n      ae_model_path: old\n      threshold: 0.05\n  layer2:\n    params:\n      threshold: 0.8\n')
            run = root / "run"
            write_json(run / "encoder/training_config.json", {"max_length": 8192})
            checkpoint = run / "autoencoder/candidate/autoencoder.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"test checkpoint")
            write_json(run / "autoencoder/selection_frozen.json", {"settings": {}, "best": {
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
                "learning_rate": 0.001, "latent_dim": 64,
                "calibration": {"threshold": 0.25}}})
            before = sha256(config)
            with patch("training.layer1_pipeline.ROOT", root):
                promote(run, before)
            self.assertIn('      threshold: 0.8', config.read_text())
            self.assertTrue((run / "previous_detection_config.yaml").exists())
            self.assertFalse(json.loads((root / "config/layer1_best.json").read_text())["test_used_for_selection"])


@unittest.skipUnless(importlib.util.find_spec("torch"), "Requires ML environment")
class AccumulationTests(unittest.TestCase):
    def test_ae_checkpoint_safe_reload_matches_errors(self):
        import torch
        from models.detectors.layer1_detector import AutoEncoder
        from training.autoencoder_training import reconstruction_errors

        torch.manual_seed(42)
        config = {"input_dim": 768, "hidden_dims": [32, 16], "latent_dim": 4, "dropout": 0.1}
        model = AutoEncoder(**config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        values = torch.randn(8, 768)
        model.train()
        reconstructed, _ = model(values)
        ((reconstructed-values)**2).mean().backward()
        optimizer.step()
        expected = reconstruction_errors(model, values.numpy(), "cpu")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "autoencoder.pth"
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                        "optimizer_parameters": optimizer.defaults, "torch_version": str(torch.__version__)}, path)
            record = torch.load(path, map_location="cpu", weights_only=True)
            loaded = AutoEncoder(**record["config"])
            loaded.load_state_dict(record["model_state_dict"])
            np.testing.assert_allclose(expected, reconstruction_errors(loaded, values.numpy(), "cpu"))

    def test_trainer_uses_normalized_accumulation(self):
        import torch
        from transformers import Trainer, TrainingArguments
        from training.modernbert_finetune import configure_loss_normalization

        class Dummy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 2)

            def forward(self, input_ids=None, labels=None, **kwargs):
                logits = self.linear(input_ids.float())
                return {"loss": torch.nn.functional.cross_entropy(logits, labels), "logits": logits}

        with tempfile.TemporaryDirectory() as directory:
            trainer = Trainer(model=configure_loss_normalization(Dummy()),
                args=TrainingArguments(output_dir=directory, use_cpu=True, report_to="none", gradient_accumulation_steps=8))
            self.assertFalse(trainer.model_accepts_loss_kwargs)


if __name__ == "__main__":
    unittest.main()
