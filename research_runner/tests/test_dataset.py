"""Temporary-file integration tests; never invoke a model, GPU, or training."""
from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from research_runner.implementation.data.data_preprocess import normalize_trace, read_jsonl, sha256, write_json, write_jsonl
from research_runner.implementation.data.data_split import SOURCES, SPLITS
from research_runner.implementation.data.prepare_training import load_prepared
from research_runner.tests.fixtures import episode
from research_runner.implementation.evaluation.asr import episode_hash, original_outcome
from research_runner.implementation.evaluation.prompt_success_judge import PROMPT_VERSION
from research_runner.dataset import prepare_dataset


class DatasetTests(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.folder = Path(contexts.enter_context(tempfile.TemporaryDirectory()))
        contexts.enter_context(redirect_stdout(io.StringIO()))
        self.subprocess = contexts.enter_context(patch(
            "research_runner.dataset.subprocess.run",
            side_effect=AssertionError("A real subprocess must not run in dataset tests")))
        contexts.enter_context(patch("research_runner.implementation.evaluation.prompt_success_judge.judge_episode",
                                side_effect=AssertionError("No model calls")))
        contexts.enter_context(patch("research_runner.implementation.evaluation.prompt_success_judge.run_vllm",
                                side_effect=AssertionError("No GPU calls")))
        self.root = self.folder / "source/data"
        self.output = self.folder / "staging/data"
        self.members = {name: [] for name in SPLITS}
        self.prompts = []
        judges = []
        for filename, kind in SOURCES.items():
            rows = []
            for index, name in enumerate(SPLITS):
                trace = episode(kind)
                trace.update(id=f"{kind or 'benign'}_{index}",
                             prompt=f"Question {kind} {index}?", label=int(kind is not None))
                if kind is None:
                    trace.update(reasoning_steps=[
                        {"node": "retrieve", "content": "Ordinary facts"},
                        {"node": "agent", "content": "Ordinary answer"}],
                        trace=["retrieve", "agent"], final_answer="Ordinary answer")
                if kind == "prompt":
                    self.prompts.append(normalize_trace(trace))
                    judges.append(self._record(normalize_trace(trace)))
                rows.append(trace)
                self.members[name].append(trace["id"])
            write_jsonl(self.root / "raw" / filename, rows)
        write_json(self.root / "splits/split_manifest.json",
                   {"episode_ids": self.members, "membership_archive": "original-reference.zip"})
        self.judges = self.folder / "source/eval/judges.jsonl"
        self.reviews = self.folder / "source/eval/reviews.jsonl"
        write_jsonl(self.judges, judges)
        write_jsonl(self.reviews, [])
        self.config = {
            "paths": {"workspace": str(Path(__file__).resolve().parents[2])},
            "data": {"root": str(self.root), "judges": str(self.judges),
                     "reviews": str(self.reviews), "reference_test_sha256": None},
            "runtime": {},
        }

    @staticmethod
    def _record(trace):
        return {"id": trace["id"], "episode_sha256": episode_hash(trace),
                "model": "meta-llama/Llama-3.1-8B-Instruct", "prompt_version": PROMPT_VERSION,
                "verdict": "FAILURE", "first_success_step": None, "reason": "Fixture refusal"}

    def _snapshot(self):
        return {p.relative_to(self.folder / "source").as_posix(): p.read_bytes()
                for p in (self.folder / "source").rglob("*") if p.is_file()}

    def _review(self, index=0, verdict="FAILURE"):
        trace = self.prompts[index]
        return {"id": trace["id"], "episode_sha256": episode_hash(trace),
                "verdict": verdict, "reason": "Explicit fixture review",
                "first_success_step": 5 if verdict == "SUCCESS" else None,
                "evidence": "done" if verdict == "SUCCESS" else ""}

    def _enable_judge(self):
        self.config["data"]["judge"] = {
            "skip": False, "backend": "vllm", "base_url": "http://127.0.0.1:9999/v1"}
        self.config["runtime"] = {"python": sys.executable, "audit_python": sys.executable}

    def test_full_preparation_preserves_inputs_membership_and_policies(self):
        before = self._snapshot()
        config = copy.deepcopy(self.config)
        manifest = prepare_dataset(self.config, self.output)
        self.assertEqual(self._snapshot(), before)
        self.assertEqual(self.config, config)
        self.subprocess.assert_not_called()
        self.assertEqual(json.loads((self.output / "dataset_manifest.json").read_text()), manifest)
        self.assertFalse((self.output / "manifest.json").exists())
        self.assertEqual(manifest["counts"]["raw_episodes"], 15)
        self.assertEqual(manifest["membership_policy"], "preserved")
        self.assertEqual(manifest["test_processed_sha256"],
                         sha256(self.output / "processed/test_processed.jsonl"))
        split_manifest = json.loads((self.output / "splits/split_manifest.json").read_text())
        self.assertEqual(split_manifest["episode_ids"], self.members)
        self.assertEqual(split_manifest["membership_archive"], "original-reference.zip")
        self.assertEqual(split_manifest["source_dir"], str(self.output / "annotated"))
        for filename in SOURCES:
            raw = read_jsonl(self.root / "raw" / filename)
            annotated = read_jsonl(self.output / "annotated" / filename)
            self.assertEqual([{k: v for k, v in row.items() if k != "asr_outcome"}
                              for row in annotated], raw)
            self.assertEqual(manifest["raw_sha256"][filename], sha256(self.root / "raw" / filename))
        for name in SPLITS:
            traces = read_jsonl(self.output / f"splits/{name}.jsonl")
            self.assertEqual([trace["id"] for trace in traces], self.members[name])
            for trace in traces:
                original_outcome(trace)
            prefixes = read_jsonl(self.output / f"processed/{name}_processed.jsonl")
            self.assertEqual(manifest["counts"]["splits"][name],
                             {"episodes": len(traces), "prefixes": len(prefixes)})
            for row in prefixes:
                onset = row["first_injection_step"]
                self.assertEqual(row["label"], int(onset is not None and row["step_number"] >= onset))
                self.assertNotIn("asr_outcome", row["sequence"])
        for stage in ("encoder", "autoencoder"):
            for name in ("train", "val"):
                rows = load_prepared(stage, name, output_dir=self.output / "training")
                self.assertTrue(all(row["attack_type"] != "correlated" for row in rows))
                self.assertEqual({row["label"] for row in rows},
                                 {0} if stage == "autoencoder" else {0, 1})
                self.assertEqual(manifest["counts"]["training"][f"{stage}/{name}.jsonl"], len(rows))
        actual_files = {p.relative_to(self.output).as_posix()
                        for p in self.output.rglob("*") if p.is_file()}
        self.assertEqual(set(manifest["artifacts"]), actual_files - {"dataset_manifest.json"})
        for relative, meta in manifest["artifacts"].items():
            path = self.output / relative
            self.assertEqual(meta["sha256"], sha256(path))
            self.assertEqual(meta["bytes"], path.stat().st_size)
            if path.suffix == ".jsonl":
                self.assertEqual(meta["rows"], len(read_jsonl(path)))
        self.assertFalse(any(p.is_symlink() for p in self.output.rglob("*")))
        self.assertFalse(any(p.name.startswith(".dataset_inputs_") for p in self.output.iterdir()))
        self.assertFalse((self.output / "raw").exists())
        self.assertEqual(manifest["asr"]["per_attack_type"]["prompt"]["failure"], 1)

    def test_final_paths_are_rebased_without_writing_to_final(self):
        final = self.folder / "published/data"
        self.config["_dataset_final"] = str(final)
        manifest = prepare_dataset(self.config, self.output)
        self.assertFalse(final.exists())
        self.assertEqual(manifest["output"], str(final))
        training = json.loads((self.output / "training/manifest.json").read_text())
        for name in ("train", "val"):
            self.assertEqual(training["sources"][name]["path"],
                             str(final / f"processed/{name}_processed.jsonl"))
        split_manifest = json.loads((self.output / "splits/split_manifest.json").read_text())
        self.assertEqual(split_manifest["source_dir"], str(final / "annotated"))
        report = json.loads((self.output / "asr/test_baseline_asr_v3.json").read_text())
        self.assertEqual(report["provenance"]["traces_path"], str(final / "splits/test.jsonl"))
        final.parent.mkdir(parents=True)
        self.output.rename(final)
        for relative, meta in manifest['artifacts'].items():
            self.assertEqual(sha256(final / relative), meta['sha256'])
            self.assertEqual((final / relative).stat().st_size, meta['bytes'])
        for stage in ("encoder", "autoencoder"):
            for name in ("train", "val"):
                self.assertTrue(load_prepared(stage, name, output_dir=final / "training"))

    def test_reference_hash_accepts_exact_processed_test_and_rejects_mismatch(self):
        initial = prepare_dataset(self.config, self.output)
        self.config["data"]["reference_test_sha256"] = initial["test_processed_sha256"]
        repeated = prepare_dataset(self.config, self.folder / "repeat")
        self.assertEqual(initial["test_processed_sha256"], repeated["test_processed_sha256"])
        self.config["data"]["reference_test_sha256"] = "0" * 64
        failed = self.folder / "mismatch"
        before = self._snapshot()
        with self.assertRaisesRegex(ValueError, "reference SHA-256 mismatch"):
            prepare_dataset(self.config, failed)
        self.assertEqual(self._snapshot(), before)
        self.assertFalse((failed / "dataset_manifest.json").exists())

    def test_missing_caches_default_to_unknown_without_subprocess(self):
        self.judges.unlink()
        self.reviews.unlink()
        before = self._snapshot()
        self.config["runtime"] = {"audit_python": sys.executable}
        result = prepare_dataset(self.config, self.output)
        self.assertTrue(result["judge"]["skip"])
        self.assertEqual(result["judge"]["pending_episodes"], 3)
        self.assertFalse(result["judge"]["subprocess_ran"])
        self.assertEqual(result["asr"]["per_attack_type"]["prompt"]["unknown"], 1)
        self.assertIsNone(result["asr"]["per_attack_type"]["prompt"]["asr"])
        self.assertEqual(self._snapshot(), before)
        self.subprocess.assert_not_called()

    def test_optional_cache_keys_never_fall_back_to_repository_inputs(self):
        del self.config["data"]["judges"]
        del self.config["data"]["reviews"]
        result = prepare_dataset(self.config, self.output)
        self.assertEqual(result["inputs"]["judges"], {"path": None, "sha256": None})
        self.assertEqual(result["judge"]["pending_episodes"], 3)
        self.assertEqual(result["asr"]["per_attack_type"]["prompt"]["unknown"], 1)

    def test_reviews_are_applied_only_to_the_staged_cache(self):
        write_jsonl(self.reviews, [self._review(verdict="SUCCESS")])
        before = self._snapshot()
        prepare_dataset(self.config, self.output)
        records = {r["id"]: r for r in read_jsonl(
            self.output / "judges/prompt_success_judgements.jsonl")}
        reviewed = records["prompt_0"]
        self.assertEqual(reviewed["verdict"], "SUCCESS")
        self.assertEqual(reviewed["llama_proposal"]["verdict"], "FAILURE")
        annotated = read_jsonl(self.output / "annotated/prompt_injection_traces.jsonl")
        self.assertEqual(annotated[0]["asr_outcome"]["verdict"], "SUCCESS")
        self.assertEqual(annotated[0]["asr_outcome"]["judge_review_status"], "assistant_trace_reviewed")
        self.assertEqual(self._snapshot(), before)

    def test_stale_review_fails_without_success_manifest_or_symlinks(self):
        write_jsonl(self.reviews, [{**self._review(), "episode_sha256": "stale"}])
        before = self._snapshot()
        with self.assertRaisesRegex(ValueError, "stale or outside source"):
            prepare_dataset(self.config, self.output)
        self.assertEqual(self._snapshot(), before)
        self.assertFalse((self.output / "dataset_manifest.json").exists())
        self.assertFalse(any(p.is_symlink() for p in self.output.rglob("*")))

    def test_explicit_judge_uses_subprocess_only_for_missing_unreviewed_episodes(self):
        self._enable_judge()
        self.config["runtime"]["python"] = str(self.folder / "not-used-python")
        write_jsonl(self.judges, [self._record(self.prompts[2])])
        write_jsonl(self.reviews, [self._review()])
        before = self._snapshot()

        def fake_run(command, *, cwd, check, env, stdout, stderr):
            self.assertEqual(stderr, subprocess.STDOUT)
            self.assertTrue(stdout.writable())
            self.assertEqual(command[0], sys.executable)
            self.assertEqual(command[1:3], ["-m", "research_runner.implementation.evaluation.prompt_success_judge"])
            self.assertEqual(cwd, Path(self.config["paths"]["workspace"]))
            self.assertTrue(check)
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(env["IG_JUDGE_GPU"], "0")
            self.assertEqual(env["IG_JUDGE_MAX_MEMORY_MIB"], "1024")
            flag = lambda name: command[command.index(name) + 1]
            self.assertEqual(flag("--backend"), "vllm")
            self.assertEqual(flag("--base-url"), "http://127.0.0.1:9999/v1")
            pending = read_jsonl(Path(flag("--traces")))
            self.assertEqual([t["id"] for t in pending], ["prompt_1"])
            self.assertEqual(read_jsonl(Path(flag("--reviews"))), [])
            for name in ("--traces", "--output", "--reviews"):
                self.assertTrue(Path(flag(name)).is_relative_to(self.output))
            cache_path = Path(flag("--output"))
            records = read_jsonl(cache_path)
            self.assertEqual({r["id"] for r in records}, {"prompt_0", "prompt_2"})
            write_jsonl(cache_path, [*records, *(self._record(trace) for trace in pending)])

        self.subprocess.side_effect = fake_run
        result = prepare_dataset(self.config, self.output)
        self.subprocess.assert_called_once()
        self.assertEqual(result["judge"], {"skip": False, "backend": "vllm",
                                          "pending_episodes": 1, "subprocess_ran": True})
        self.assertEqual(self._snapshot(), before)
        self.assertFalse(any(p.is_symlink() for p in self.output.rglob("*")))

    def test_explicit_judge_with_all_cached_or_reviewed_does_not_spawn(self):
        self._enable_judge()
        self.config["runtime"] = {"device": "cpu"}
        records = read_jsonl(self.judges)
        records[0]["model"] = "different-legacy-model"
        records[0]["prompt_version"] = "legacy_success_judge_user_accepted_20261021"
        write_jsonl(self.judges, records[:-1])
        write_jsonl(self.reviews, [self._review(index=2)])
        result = prepare_dataset(self.config, self.output)
        self.assertEqual(result["judge"]["pending_episodes"], 0)
        self.subprocess.assert_not_called()

    def test_judge_preserves_virtualenv_interpreter_and_handles_null_base_url(self):
        self._enable_judge()
        interpreter = self.folder / ".venv/bin/python"
        interpreter.parent.mkdir(parents=True)
        interpreter.symlink_to(sys.executable)
        self.config["runtime"]["audit_python"] = str(interpreter)
        self.config["data"]["judge"]["base_url"] = None
        write_jsonl(self.judges, [])

        def fake_run(command, **kwargs):
            self.assertEqual(command[0], str(interpreter))
            self.assertEqual(command[command.index("--base-url") + 1], "http://127.0.0.1:11434/v1")
            write_jsonl(Path(command[command.index("--output") + 1]),
                        [self._record(trace) for trace in self.prompts])

        self.subprocess.side_effect = fake_run
        result = prepare_dataset(self.config, self.output)
        self.assertTrue(result["judge"]["subprocess_ran"])
        self.subprocess.assert_called_once()

    def test_judging_missing_records_requires_explicit_runtime(self):
        self.config["data"]["judge"] = {"skip": False}
        write_jsonl(self.judges, [])
        with self.assertRaisesRegex(ValueError, "explicit runtime"):
            prepare_dataset(self.config, self.output)
        self.subprocess.assert_not_called()
        self.assertFalse((self.output / "dataset_manifest.json").exists())

    def test_judge_forwards_configured_gpu_and_memory_to_subprocess(self):
        self._enable_judge()
        write_jsonl(self.judges, [])
        self.config["runtime"]["max_gpu_memory_mib"] = 1536
        for device, expected in (("cuda", "0"), ("cuda:2", "2"), ("3", "3")):
            with self.subTest(device=device):
                self.config["runtime"]["device"] = device

                def fake_run(command, *, env, **kwargs):
                    self.assertEqual(env["IG_JUDGE_GPU"], expected)
                    self.assertEqual(env["IG_JUDGE_MAX_MEMORY_MIB"], "1536")
                    write_jsonl(Path(command[command.index("--output") + 1]),
                                [self._record(trace) for trace in self.prompts])

                self.subprocess.side_effect = fake_run
                with patch.dict("os.environ", {"IG_JUDGE_GPU": "99", "IG_JUDGE_MAX_MEMORY_MIB": "1"}):
                    result = prepare_dataset(self.config, self.folder / f"gpu-{expected}-{device}")
                self.assertTrue(result["judge"]["subprocess_ran"])
        self.assertEqual(self.subprocess.call_count, 3)

    def test_pending_vllm_judging_rejects_cpu_without_spawning(self):
        self._enable_judge()
        self.config["runtime"]["device"] = "cpu"
        write_jsonl(self.judges, [])
        before = self._snapshot()
        with self.assertRaisesRegex(ValueError, "requires a GPU"):
            prepare_dataset(self.config, self.output)
        self.assertEqual(self._snapshot(), before)
        self.subprocess.assert_not_called()
        self.assertFalse((self.output / "dataset_manifest.json").exists())
        self.assertFalse(any(p.is_symlink() for p in self.output.rglob("*")))

    def test_openai_compatible_judging_does_not_require_a_local_gpu(self):
        self._enable_judge()
        self.config["runtime"]["device"] = "cpu"
        self.config["data"]["judge"]["backend"] = "openai-compatible"
        write_jsonl(self.judges, [])

        def fake_run(command, **kwargs):
            write_jsonl(Path(command[command.index("--output") + 1]),
                        [self._record(trace) for trace in self.prompts])

        self.subprocess.side_effect = fake_run
        result = prepare_dataset(self.config, self.output)
        self.assertTrue(result["judge"]["subprocess_ran"])
        self.subprocess.assert_called_once()

    def test_failed_judge_propagates_without_modifying_inputs(self):
        self._enable_judge()
        del self.config["runtime"]["audit_python"]
        self.config["data"]["judge"]["backend"] = "openai-compatible"
        write_jsonl(self.judges, [])
        before = self._snapshot()
        self.subprocess.side_effect = subprocess.CalledProcessError(1, ["mock-judge"])
        with self.assertRaises(subprocess.CalledProcessError):
            prepare_dataset(self.config, self.output)
        self.assertEqual(self.subprocess.call_args.args[0][0], sys.executable)
        self.assertEqual(self._snapshot(), before)
        self.assertFalse((self.output / "dataset_manifest.json").exists())
        self.assertFalse(any(p.is_symlink() for p in self.output.rglob("*")))

    def test_membership_can_be_recovered_from_splits_or_processed_prefixes(self):
        (self.root / "splits/split_manifest.json").unlink()
        for use_processed in (False, True):
            with self.subTest(processed=use_processed):
                for name, ids in self.members.items():
                    path = self.root / f"splits/{name}.jsonl"
                    if use_processed:
                        path.unlink()
                        write_jsonl(self.root / f"processed/{name}_processed.jsonl",
                                    [{"id": f"{id_}_step{step}", "original_trace_id": id_}
                                     for id_ in ids for step in (0, 1)])
                    else:
                        write_jsonl(path, [{"id": id_} for id_ in ids])
                before = self._snapshot()
                output = self.folder / f"recovered-{use_processed}"
                prepare_dataset(self.config, output)
                manifest = json.loads((output / "splits/split_manifest.json").read_text())
                self.assertEqual(manifest["episode_ids"], self.members)
                self.assertEqual(self._snapshot(), before)

    def test_missing_or_invalid_membership_is_not_silently_resplit(self):
        path = self.root / "splits/split_manifest.json"
        path.unlink()
        with self.assertRaisesRegex(ValueError, "refusing to resplit"):
            prepare_dataset(self.config, self.output)
        self.assertFalse(self.output.exists())
        write_json(path, {"episode_ids": {**self.members, "test": self.members["train"]}})
        with self.assertRaisesRegex(ValueError, "missing, extra, or overlapping"):
            prepare_dataset(self.config, self.output)
        self.assertFalse((self.output / "dataset_manifest.json").exists())

    def test_artifact_store_publishes_its_own_root_manifest_without_conflict(self):
        from research_runner.artifacts import ArtifactStore

        results = self.folder / "results"
        final = results / "dataset"
        self.config["_dataset_final"] = str(final)
        store = ArtifactStore(results, self.config)
        signature = "fixture-dataset-signature"
        with patch("research_runner.artifacts._source_commit", return_value=None), \
                store.transaction("dataset", signature) as stage:
            report = prepare_dataset(self.config, stage)
            write_json(stage / "dataset_report.json", report)
            self.assertFalse((stage / "manifest.json").exists())
            self.assertTrue((stage / "dataset_manifest.json").is_file())
        self.assertEqual(store.load("dataset", signature), final)
        stored = json.loads((final / "manifest.json").read_text())
        self.assertEqual(stored["status"], "completed")
        self.assertIsNone(stored["source_commit"])
        self.assertIn("dataset_manifest.json", stored["files"])
        self.assertEqual(json.loads((final / "dataset_manifest.json").read_text()), report)
        for relative, meta in report["artifacts"].items():
            self.assertEqual(sha256(final / relative), meta["sha256"])
        self.assertTrue(load_prepared("encoder", "train", output_dir=final / "training"))
        self.subprocess.assert_not_called()

    def test_rejects_unsafe_or_nonempty_output_and_unresolved_paths(self):
        for output in (self.root, self.root / "new-output", self.root.parent,
                       self.judges.parent):
            with self.subTest(output=output), self.assertRaises(ValueError):
                prepare_dataset(self.config, output)
        self.output.mkdir(parents=True)
        write_json(self.output / "existing.json", {"keep": True})
        with self.assertRaisesRegex(ValueError, "absent or empty"):
            prepare_dataset(self.config, self.output)
        self.assertEqual(json.loads((self.output / "existing.json").read_text()), {"keep": True})
        config = copy.deepcopy(self.config)
        config["data"]["root"] = "relative/data"
        with self.assertRaisesRegex(ValueError, "absolute"):
            prepare_dataset(config, self.folder / "unused")

    def test_invalid_reference_or_judge_options_fail_closed(self):
        self.config["data"]["reference_test_sha256"] = "invalid"
        with self.assertRaisesRegex(ValueError, "SHA-256 digest"):
            prepare_dataset(self.config, self.output)
        self.config["data"]["reference_test_sha256"] = None
        self.config["data"]["judge"] = {"skip": "false"}
        with self.assertRaisesRegex(ValueError, "boolean"):
            prepare_dataset(self.config, self.output)
        self.subprocess.assert_not_called()


if __name__ == "__main__":
    unittest.main()
