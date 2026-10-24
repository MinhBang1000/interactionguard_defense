"""Temporary-directory tests for atomic results and credential-safe backups."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from research_runner.artifacts import (
    ArtifactError, ArtifactStore, ExistingResultsError, IncompatibleResultsError,
    backup_snapshot, file_sha256, fingerprint, read_json, read_jsonl, write_json, write_jsonl,
)
from research_runner.config import load_config


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.config = {"version": 1,
                       "paths": {"workspace": str(self.root), "results": str(self.root / "results"),
                                 "backup": str(self.root / "backup")},
                       "backup": {"include_checkpoints": False},
                       "models": {"audit": {"api_key_env": "EXAMPLE_API_KEY"}}}
        self.store = ArtifactStore(self.root / "results", self.config)

    def create(self, name="configuration_stage1", value=1, signature="signature"):
        with self.store.transaction(name, signature) as stage:
            write_json(stage / "data.json", {"value": value})
            write_json(stage / "stage_complete.json", {"completed": True})
        return self.store.load(name, signature)

    def snapshot(self, path):
        return {file.relative_to(path).as_posix(): file.read_bytes()
                for file in path.rglob("*") if file.is_file()}

    def test_json_helpers_and_stable_fingerprint(self):
        path = self.root / "nested/value.json"
        write_json(path, {"value": 7})
        self.assertEqual(read_json(path), {"value": 7})
        write_jsonl(path, ({"i": i} for i in range(3)))
        self.assertEqual(read_jsonl(path), [{"i": i} for i in range(3)])
        self.assertEqual(file_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(fingerprint({"a": 1, "b": 2}), fingerprint({"b": 2, "a": 1}))
        with self.assertRaises(ValueError):
            fingerprint(float("nan"))

    def test_failed_jsonl_write_preserves_previous_file(self):
        path = self.root / "data.jsonl"
        write_jsonl(path, [1])

        def rows():
            yield 2
            raise RuntimeError("interrupted generator")

        with self.assertRaises(RuntimeError):
            write_jsonl(path, rows())
        self.assertEqual(read_jsonl(path), [1])
        self.assertEqual(list(self.root.glob(".*.tmp")), [])

    def test_publish_manifest_and_effective_config(self):
        self.assertFalse(self.store.root.exists())
        with self.store.transaction("configuration_stage1", "signature") as stage:
            self.assertFalse((self.store.root / "configuration_stage1").exists())
            write_json(stage / "data.json", {"value": 1})
        target = self.store.load("configuration_stage1", "signature")
        self.assertFalse(stage.exists())
        manifest = read_json(target / "manifest.json")
        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["signature"], "signature")
        self.assertEqual(set(manifest["files"]), {"data.json", "effective_config.json"})
        self.assertEqual(read_json(target / "effective_config.json"), self.config)
        self.assertIn("source_commit", manifest)

    def test_existing_results_default_no_before_mutation(self):
        target = self.create()
        before = self.snapshot(self.store.root)
        with self.assertRaises(ExistingResultsError):
            with self.store.transaction("configuration_stage1", "signature"):
                self.fail("Existing results must fail before caller runs")
        self.assertEqual(before, self.snapshot(self.store.root))
        self.assertEqual(read_json(target / "data.json"), {"value": 1})

    def test_caller_failure_preserves_previous_results(self):
        target = self.create()
        before = self.snapshot(target)
        with self.assertRaisesRegex(RuntimeError, "training failed"):
            with self.store.transaction("configuration_stage1", "new", overwrite=True) as stage:
                write_json(stage / "data.json", {"value": 2})
                raise RuntimeError("training failed")
        self.assertEqual(self.snapshot(target), before)
        self.assertEqual(list((self.store.root / ".staging").iterdir()), [])

    def test_failed_worker_logs_are_retained_but_never_completed_or_resumable(self):
        target = self.create()
        before = self.snapshot(target)
        with self.assertRaisesRegex(RuntimeError, "worker failed"):
            with self.store.transaction("configuration_stage1", "new", overwrite=True) as stage:
                worker = stage / "worker"
                worker.mkdir()
                (worker / "worker.log").write_text("worker failed\napi_key=fixture-secret\n", encoding="utf-8")
                (worker / "escape.log").symlink_to(target / "data.json")
                (worker / "partial.pth").write_bytes(b"incomplete weights")
                raise RuntimeError("worker failed")
        self.assertEqual(self.snapshot(target), before)
        attempts = list((self.store.root / ".failed").iterdir())
        self.assertEqual(len(attempts), 1)
        attempt = attempts[0]
        manifest = read_json(attempt / "manifest.json")
        self.assertEqual(manifest["status"], "failed")
        log = (attempt / "logs/worker/worker.log").read_text(encoding="utf-8")
        self.assertIn("worker failed", log)
        self.assertNotIn("fixture-secret", log)
        self.assertFalse(any(file.is_symlink() or file.suffix == ".pth" for file in attempt.rglob("*")))
        for relative, expected in manifest["files"].items():
            self.assertEqual(file_sha256(attempt / relative), expected)
        with self.assertRaises(ArtifactError):
            self.store.load(f".failed/{attempt.name}")

    def test_diagnostic_failure_never_masks_worker_failure(self):
        with patch.object(self.store, "_retain_failure", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(RuntimeError, "worker failed"):
                with self.store.transaction("broken", "signature"):
                    raise RuntimeError("worker failed")
        self.assertFalse((self.store.root / "broken").exists())

    def test_interrupted_atomic_publish_restores_previous_results(self):
        target = self.create()
        before = self.snapshot(target)
        import os
        replace = os.replace

        def interrupted(source, destination):
            if Path(source).parent.name == ".staging" and Path(destination) == target:
                raise KeyboardInterrupt("publish interrupted")
            return replace(source, destination)

        with patch("research_runner.artifacts.os.replace", side_effect=interrupted):
            with self.assertRaises(KeyboardInterrupt):
                with self.store.transaction("configuration_stage1", "new", overwrite=True) as stage:
                    write_json(stage / "data.json", {"value": 2})
        self.assertEqual(self.snapshot(target), before)
        self.store.load("configuration_stage1", "signature")
        self.assertEqual(list(target.parent.glob(".*.previous-*")), [])

    def test_explicit_overwrite_replaces_old_files(self):
        target = self.create()
        with self.store.transaction("configuration_stage1", "new", overwrite=True) as stage:
            write_json(stage / "replacement.json", True)
        self.assertEqual(self.store.load("configuration_stage1", "new"), target)
        self.assertFalse((target / "data.json").exists())

    def test_completed_resume_is_read_only_and_does_not_skip_stage2(self):
        target = self.create()
        before = self.snapshot(target)
        with self.store.transaction("configuration_stage1", "signature", resume=True) as resumed:
            self.assertEqual(resumed, target)
        self.assertEqual(self.snapshot(target), before)
        with self.store.transaction("configuration_stage2", "signature", resume=True) as stage:
            self.assertNotEqual(stage, target)
            self.assertFalse((stage / "stage_complete.json").exists())
            write_json(stage / "stage_complete.json", {"stage": 2})
        self.store.load("configuration_stage2", "signature")

    def test_resume_rejects_incompatible_or_corrupt_results(self):
        target = self.create()
        with self.assertRaises(IncompatibleResultsError):
            with self.store.transaction("configuration_stage1", "new", resume=True):
                self.fail("Incompatible resume")
        write_json(target / "data.json", {"value": "tampered"})
        with self.assertRaises(ArtifactError):
            with self.store.transaction("configuration_stage1", "signature", resume=True):
                self.fail("Corrupt resume")

    def test_stale_missing_and_added_file_hashes_are_rejected(self):
        for change in ("modify", "delete", "add"):
            with self.subTest(change=change):
                target = self.create(name=change)
                if change == "modify":
                    write_json(target / "data.json", {"value": 8})
                elif change == "delete":
                    (target / "data.json").unlink()
                else:
                    write_json(target / "extra.json", {})
                with self.assertRaises(ArtifactError):
                    self.store.load(change)

    def test_manifest_paths_status_and_signature_are_validated(self):
        for change in ({"status": "running"}, {"signature": None},
                       {"files": {"../outside": "a" * 64}}, {"files": {"/absolute": "a" * 64}}):
            with self.subTest(change=change):
                name = "case-" + fingerprint(change)[:8]
                target = self.create(name=name)
                manifest = read_json(target / "manifest.json")
                manifest.update(change)
                write_json(target / "manifest.json", manifest)
                with self.assertRaises(ArtifactError):
                    self.store.load(name)

    def test_unsafe_names_and_symlinks_are_rejected(self):
        for name in ("../escape", "/absolute", "a/../b", "a\\b", "a//b", ".locks"):
            with self.subTest(name=name), self.assertRaises(ArtifactError):
                with self.store.transaction(name, "signature"):
                    self.fail("Unsafe name")
        secret = self.root / "outside.json"
        write_json(secret, {"secret": True})
        with self.assertRaises(ArtifactError):
            with self.store.transaction("linked", "signature") as stage:
                (stage / "linked.json").symlink_to(secret)
        self.assertFalse((self.store.root / "linked").exists())

    def test_preflight_and_mutually_exclusive_controls(self):
        self.assertIsNone(self.store.ensure("new", "signature"))
        self.assertFalse(self.store.root.exists())
        target = self.create()
        self.assertEqual(self.store.ensure("configuration_stage1", "signature", resume=True), target)
        with self.assertRaises(ValueError):
            with self.store.transaction("configuration_stage1", "signature", resume=True, overwrite=True):
                self.fail("Conflicting controls")

    def test_internal_directories_cannot_redirect_writes_outside_store(self):
        outside = self.root / "outside"
        outside.mkdir()
        self.store.root.mkdir()
        (self.store.root / ".staging").symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ArtifactError):
            with self.store.transaction("experiment", "signature"):
                self.fail("Must reject redirected staging")
        self.assertEqual(list(outside.iterdir()), [])

    def test_git_head_snapshot_uses_files_not_subprocess(self):
        git = self.root / ".git"
        git.mkdir()
        (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        (git / "packed-refs").write_text("a" * 40 + " refs/heads/main\n", encoding="utf-8")
        with patch("subprocess.run", side_effect=AssertionError("No subprocess")):
            target = self.create()
        self.assertEqual(read_json(target / "manifest.json")["source_commit"], "a" * 40)

    def test_zip_filters_and_verifies_every_member(self):
        self.create()
        results = self.store.root
        for relative in (".env", ".env.production", "secrets/key.json", ".git/config", "cache/data.json",
                         "__pycache__/data.pyc", "env/lib/site.py", "private.key", "training/model.pth",
                         "training/model.safetensors", ".old.previous-abcd/data.json"):
            path = results / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("MUST-NOT-BE-IN-BACKUP", encoding="utf-8")
        outside = self.root / "outside.json"
        write_json(outside, {"secret": True})
        (results / "linked.json").symlink_to(outside)
        (results / "linked-dir").symlink_to(self.root, target_is_directory=True)
        with zipfile.ZipFile(results / "previous.zip", "w") as archive:
            archive.writestr("old", "old")
        with zipfile.ZipFile(results / "disguised.data", "w") as archive:
            archive.writestr("old", "old")
        self.config["models"]["audit"]["api_key"] = "never archive this literal"
        backup = self.store.backup()
        second = self.store.backup()
        self.assertNotEqual(backup, second)
        self.assertEqual(backup.parent, self.root / "backup")
        with zipfile.ZipFile(second) as archive:
            self.assertIsNone(archive.testzip())
            names = archive.namelist()
            self.assertIn("results/configuration_stage1/data.json", names)
            self.assertIn("effective_config.json", names)
            self.assertFalse(any("MUST-NOT-BE-IN-BACKUP" in archive.read(name).decode("utf-8") for name in names))
            self.assertFalse(any(name.endswith(".zip") or "linked" in name or name.startswith("/") or ".." in name.split("/") for name in names))
            self.assertNotIn("results/disguised.data", names)
            self.assertNotIn("api_key", json.loads(archive.read("effective_config.json"))["models"]["audit"])
            checksums = json.loads(archive.read("checksums.json"))["files"]
            self.assertEqual(set(checksums), set(names) - {"checksums.json"})
            for name, expected in checksums.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)

    def test_checkpoint_and_extra_path_inclusion_is_config_driven(self):
        checkpoint = self.root / "models/encoder/model.safetensors"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"checkpoint")
        self.config["models"]["encoder"] = {"checkpoint_path": str(checkpoint.parent)}
        self.config["backup"]["include_checkpoints"] = True
        extra = self.root / "notes.txt"
        extra.write_text("notes", encoding="utf-8")
        self.config["backup"]["include_paths"] = [str(extra)]
        with zipfile.ZipFile(backup_snapshot(self.config)) as archive:
            self.assertIn("checkpoints/encoder/checkpoint_path/model.safetensors", archive.namelist())
            self.assertTrue(any(name.startswith("additional/") for name in archive.namelist()))

    def test_zip_serialized_checkpoints_require_opt_in_but_other_archives_stay_excluded(self):
        training = self.store.root / "training"
        training.mkdir(parents=True)
        checkpoint_names = ["fake.pt", "autoencoder.pth", "weights.bin", "model.ckpt"]
        for name in checkpoint_names + ["previous.zip", "renamed.data"]:
            with zipfile.ZipFile(training / name, "w") as archive:
                archive.writestr("archive/data.pkl", b"fake serialized weights")
        for enabled in (False, True):
            with self.subTest(include_checkpoints=enabled):
                self.config["backup"]["include_checkpoints"] = enabled
                with zipfile.ZipFile(self.store.backup()) as archive:
                    checksums = json.loads(archive.read("checksums.json"))["files"]
                    self.assertIsNone(archive.testzip())
                    for name in checkpoint_names:
                        member = f"results/training/{name}"
                        self.assertEqual(member in archive.namelist(), enabled)
                        if enabled:
                            self.assertEqual(archive.read(member), (training / name).read_bytes())
                            self.assertEqual(checksums[member], file_sha256(training / name))
                    self.assertNotIn("results/training/previous.zip", archive.namelist())
                    self.assertNotIn("results/training/renamed.data", archive.namelist())

    def test_selected_secret_root_is_not_archived(self):
        path = self.root / "secrets" / "credentials.json"
        write_json(path, {"credential": "never"})
        self.config["backup"]["include_paths"] = [str(path.parent)]
        with zipfile.ZipFile(self.store.backup()) as archive:
            self.assertFalse(any(name.startswith("additional/") for name in archive.namelist()))

    def test_backup_inside_results_does_not_include_previous_archives(self):
        self.create()
        self.config["paths"]["backup"] = str(self.store.root / "backup")
        previous = self.store.backup()
        with zipfile.ZipFile(self.store.backup()) as archive:
            self.assertFalse(any("backup/" in name or previous.name in name for name in archive.namelist()))

    def test_failed_backup_crc_does_not_publish_or_remove_previous_zip(self):
        previous = self.store.backup()
        before = previous.read_bytes()
        with patch("research_runner.artifacts.zipfile.ZipFile.testzip", return_value="damaged"):
            with self.assertRaises(ArtifactError):
                self.store.backup()
        self.assertEqual(previous.read_bytes(), before)
        self.assertEqual(list(previous.parent.glob("*.zip")), [previous])
        self.assertEqual(list(previous.parent.glob("*.tmp")), [])

    def test_disabled_backup_has_no_side_effects(self):
        self.config["backup"]["enabled"] = False
        self.assertIsNone(self.store.backup())
        self.assertFalse((self.root / "backup").exists())

    def test_backup_includes_original_yaml_and_dataset_effective_snapshot(self):
        path = self.root / "campaign.yaml"
        path.write_text("version: 1\npaths:\n  results: results\n  backup: backup\n", encoding="utf-8")
        config = load_config(path, root=self.root)
        store = ArtifactStore(config["paths"]["results"], config)
        with store.transaction("dataset", "signature") as stage:
            write_json(stage / "dataset_manifest.json", {})
        with zipfile.ZipFile(store.backup()) as archive:
            self.assertEqual(archive.read("source_config.yaml"), path.read_bytes())
            self.assertIn("results/dataset/effective_config.json", archive.namelist())
            checksums = json.loads(archive.read("checksums.json"))["files"]
            self.assertEqual(checksums["source_config.yaml"], file_sha256(path))
        path.write_text("version: 1\n# changed after loading\n", encoding="utf-8")
        with self.assertRaisesRegex(ArtifactError, "changed after"):
            store.backup()


if __name__ == "__main__":
    unittest.main()
