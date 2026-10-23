"""CPU campaign integration tests through the real engine and worker dispatcher.

Only model methods are fake: no engine, dataset, routing, calibration, reporting,
artifact publication, or backup implementation is mocked.
"""
from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from data.data_preprocess import normalize_trace
from data.data_split import SOURCES, SPLITS
from data.test_preprocessing import episode
from eval.asr import episode_hash
from eval.prompt_success_judge import PROMPT_VERSION
from research_runner.artifacts import (
    ArtifactError, ExistingResultsError, IncompatibleResultsError,
    file_sha256, read_json, read_jsonl, write_json, write_jsonl,
)
from research_runner.config import load_config
from research_runner.engine import ResearchRunner
from research_runner.payloads import build_payload
from research_runner import worker


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/experiments.yaml"


class FakeBackend:
    """Deterministic model stand-ins with the production adapter's public API."""

    def __init__(self, config):
        self.config = config
        self.calls = []
        self.bad_score_ids = False
        self.bad_response_ids = False
        self.fail_next_inference = False

    def train(self, seed, output):
        output = Path(output)
        self.calls.append(("train", seed))
        encoder = output / "encoder"
        ae = output / "autoencoder.pth"
        selection = output / "selection.json"
        write_json(encoder / "config.json", {"fake_model": True, "seed": seed})
        write_json(encoder / "model.safetensors", {"fake_weights": seed})
        write_json(ae, {"fake_weights": seed})
        metadata = {"seed": seed, "l1_threshold": 0.5,
                    "encoder_path": str(encoder), "ae_path": str(ae),
                    "selection_path": str(selection)}
        write_json(selection, {"threshold": 0.5, "selection_source": "validation_only"})
        write_json(output / "checkpoint.json", metadata)
        return metadata

    def score_l1(self, rows, checkpoint):
        self.calls.append(("score_l1", checkpoint["seed"], tuple(r["id"] for r in rows)))
        records = [{"id": row["id"], "score": 0.8 if row["label"] else 0.1,
                    "seconds": 0.001, "timing_scope": "fake_cpu_per_prefix"}
                   for row in rows]
        if self.bad_score_ids and records:
            records[0]["id"] = "foreign_score_id"
        return records

    def score_detector(self, rows, model_config, channels=None):
        self.calls.append(("score_detector", model_config["model"],
                           tuple(channels or ()), tuple(r["id"] for r in rows)))
        # Benign validation scores fix a nontrivial gate. Some positive prefixes
        # remain below it, so the actual routing invokes L3 rather than going idle.
        return [{"id": row["id"],
                 "score": (0.9 if row["step_number"] % 2 else 0.1) if row["label"] else 0.2,
                 "channels": [{"channel": c, "score": 0.2} for c in channels or ("prompt", "retrieval", "tool")],
                 "seconds": 0.002, "model": model_config["model"],
                 "timing_scope": "fake_cpu_per_prefix"} for row in rows]

    def score_rag(self, rows, train, settings):
        self.calls.append(("score_rag", tuple(r["id"] for r in train), settings["alpha"]))
        return [{"id": row["id"], "score": 0.7 if row["label"] else 0.1,
                 "prediction": int(bool(row["channels"]["retrieval"]) and bool(row["label"])),
                 "seconds": 0.004} for row in rows]

    def infer_l3(self, requests, backend_config=None):
        self.calls.append(("infer_l3", deepcopy(backend_config), deepcopy(requests)))
        if self.fail_next_inference:
            self.fail_next_inference = False
            raise RuntimeError("Synthetic interrupted L3 worker")
        responses = []
        for request in requests:
            prediction = int(any(text in request["payload"] for text in
                                 ("IMPORTANT:", "'get_information' tool")))
            responses.append({"request_id": request["request_id"], "prediction": prediction,
                              "seconds": 0.003, "cache_used": False,
                              "timing_scope": "fake_cpu_uncached_request",
                              "backend": (backend_config or self.config["models"]["l3"])["backend"]})
        if self.bad_response_ids and responses:
            responses[0]["request_id"] = "foreign_request_id"
        # Deliberately return out of order; production alignment must join by ID.
        return list(reversed(responses))


class InlineExecutor:
    """Synchronous CPU executor retaining the production worker protocol."""

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend
        self.calls = []
        self.active = False

    def run(self, operation, job, destination, *, llm=False):
        if self.active:
            raise AssertionError("Worker phases must not overlap")
        if llm != (operation == "infer"):
            raise AssertionError("Only inference may use the LLM worker")
        self.calls.append({"operation": operation, "job": deepcopy(job),
                           "destination": str(destination), "llm": llm})
        self.active = True
        try:
            return worker.execute(operation, self.config, job, destination, backend=self.backend)
        finally:
            self.active = False


class EngineTests(unittest.TestCase):
    def setUp(self):
        contexts = ExitStack()
        self.addCleanup(contexts.close)
        self.workspace = Path(contexts.enter_context(tempfile.TemporaryDirectory()))
        contexts.enter_context(redirect_stdout(io.StringIO()))
        contexts.enter_context(patch("subprocess.run", side_effect=AssertionError("No real subprocess in CPU integration tests")))
        contexts.enter_context(patch("eval.prompt_success_judge.judge_episode", side_effect=AssertionError("No model calls")))
        contexts.enter_context(patch("eval.prompt_success_judge.run_vllm", side_effect=AssertionError("No GPU calls")))
        self.config = load_config(CONFIG, root=self.workspace)
        self.config["runtime"].update(device="cpu", python=sys.executable, l3_python=sys.executable)
        self.config["training"]["seeds"] = [42, 43]
        self.config["data"]["reference_test_sha256"] = None
        self.config["data"]["judge"]["skip"] = True
        self.config["experiments"]["benign"]["total_size"] = 5
        self.config["experiments"]["timing"]["generate_figures"] = False
        self.results = Path(self.config["paths"]["results"])
        self.members = {split: [] for split in SPLITS}
        judges = []
        for filename, kind in SOURCES.items():
            raw = []
            for index, split in enumerate(SPLITS):
                trace = episode(kind)
                trace.update(id=f"{kind or 'benign'}_{index}",
                             prompt=f"Fixture question {kind} {index}?", label=int(kind is not None))
                if kind is None:
                    trace.update(reasoning_steps=[{"node": "retrieve", "content": "Ordinary facts"},
                                                  {"node": "agent", "content": "Ordinary answer"}],
                                 trace=["retrieve", "agent"], final_answer="Ordinary answer")
                if kind == "prompt":
                    judges.append({"id": trace["id"], "episode_sha256": episode_hash(normalize_trace(trace)),
                                   "model": "meta-llama/Llama-3.1-8B-Instruct", "prompt_version": PROMPT_VERSION,
                                   "verdict": "FAILURE", "first_success_step": None, "reason": "Fixture refusal"})
                raw.append(trace)
                self.members[split].append(trace["id"])
            write_jsonl(Path(self.config["data"]["root"]) / "raw" / filename, raw)
        write_json(Path(self.config["data"]["root"]) / "splits/split_manifest.json",
                   {"episode_ids": self.members, "membership_archive": "fixture-membership.zip"})
        write_jsonl(self.config["data"]["judges"], judges)
        write_jsonl(self.config["data"]["reviews"], [])
        self.originals = [Path(self.config["data"]["root"]), self.workspace / "eval"]
        self.source_snapshot = self.snapshot_sources()
        self.real_config_hash = file_sha256(CONFIG)
        self.backend = FakeBackend(self.config)
        self.executor = InlineExecutor(self.config, self.backend)
        self.runner = ResearchRunner(self.config, executor=self.executor)

    def tearDown(self):
        self.assertEqual(self.snapshot_sources(), self.source_snapshot, "Original dataset or judges were modified")
        self.assertEqual(file_sha256(CONFIG), self.real_config_hash, "Repository config was modified")

    def snapshot_sources(self):
        return {str(path.relative_to(self.workspace)): file_sha256(path)
                for directory in self.originals for path in directory.rglob("*") if path.is_file()}

    def prepare(self, training=False, full=False):
        self.runner.run(0)
        if training or full:
            self.runner.run(1)
        if full:
            self.runner.run(2)

    def verify_artifact(self, name):
        path = self.runner.store.load(name)
        manifest = read_json(path / "manifest.json")
        self.assertEqual(manifest["status"], "completed")
        for relative, digest in manifest["files"].items():
            self.assertEqual(file_sha256(path / relative), digest)
        return path

    def verify_report(self, directory, expected_ids=None, episode_level=True):
        decisions = read_jsonl(directory / "prefix_decisions.jsonl")
        ids = [decision["id"] for decision in decisions]
        self.assertEqual(len(ids), len(set(ids)))
        if expected_ids is not None:
            self.assertEqual(ids, expected_ids)
        self.assertTrue(all(type(d["prediction"]) is int and d["prediction"] in (0, 1) for d in decisions))
        self.assertTrue(all(not d.get("pending_l3", False) for d in decisions))
        stats = read_json(directory / "metrics.json")
        self.assertEqual(stats["samples"], len(ids))
        self.assertEqual(stats["sample_ids"], ids)
        self.assertEqual(sum(stats[key] for key in ("tp", "fp", "fn", "tn")), len(ids))
        timing = read_json(directory / "timing.json")
        self.assertEqual(timing["interpretation"], "component_based_pipeline_estimate_not_direct_end_to_end")
        self.assertGreaterEqual(timing["estimated_pipeline_seconds_per_sample"], 0)
        for layer in ("L1", "L2", "L3"):
            self.assertEqual(stats["processed_counts"][layer],
                             sum(any(s["layer"] == layer for s in d.get("route", [])) for d in decisions))
        if episode_level:
            report = read_json(directory / "asr.json")
            self.assertEqual(report["episodes"], read_jsonl(directory / "episode_results.jsonl"))
            self.assertEqual({record["id"] for record in report["episodes"]}, set(self.members["test"]))
            self.assertEqual(report["protocol"], "offline_replay_not_live_execution")
        else:
            self.assertFalse((directory / "asr.json").exists())
        return decisions

    def test_menu_zero_through_eight_complete_campaign(self):
        self.prepare(training=True)
        dataset = self.verify_artifact("dataset")
        training = self.verify_artifact("training")
        self.assertEqual(read_json(dataset / "splits/split_manifest.json")["episode_ids"], self.members)
        checkpoints = read_json(training / "checkpoints.json")
        self.assertEqual(set(checkpoints), {"42", "43"})
        for checkpoint in checkpoints.values():
            self.assertEqual(checkpoint["l1_threshold"], 0.5)
            self.assertTrue(Path(checkpoint["encoder_path"]).is_dir())
            self.assertTrue(Path(checkpoint["ae_path"]).is_file())
            self.assertNotIn(".staging", checkpoint["encoder_path"])
        self.runner.run(2)
        first = self.verify_artifact("configuration_stage1")
        second = self.verify_artifact("configuration_stage2")
        rows = self.runner.rows()
        ids = [r["id"] for r in rows]
        for seed in (42, 43):
            requests = read_jsonl(first / f"seed_{seed}/l3_requests.jsonl")
            self.assertTrue(requests)
            for order in self.config["experiments"]["configuration"]["orders"]:
                staged = read_jsonl(first / f"seed_{seed}/{order}/routing.jsonl")
                self.assertEqual([d["id"] for d in staged], ids)
                pending = {d["id"] for d in staged if d["pending_l3"]}
                self.assertEqual(pending, {r["prefix_id"] for r in requests if r["configuration"] == order})
                self.verify_report(second / f"seed_{seed}/{order}", ids)
            for request in requests:
                source = next(r for r in rows if r["id"] == request["prefix_id"])
                self.assertEqual(request["payload"], build_payload(source["sequence"], self.config["models"]["l3"]["payload"]))
                self.assertEqual(request["request_id"], hashlib.sha256(request["payload"].encode()).hexdigest())
            timing = read_json(second / f"seed_{seed}/l3/timing.json")
            self.assertTrue(timing["uncached_measured"])
            self.assertFalse(timing["cache_used"])
            self.assertEqual(timing["submitted_requests"], len(requests))
            responses = read_jsonl(second / f"seed_{seed}/l3/aligned_responses.jsonl")
            self.assertEqual([r["prefix_id"] for r in responses], [r["prefix_id"] for r in requests])
            self.assertTrue(all(r["seconds"] == 0.003 for r in responses))
        for item, name in ((3, "baselines"), (4, "channel_ablation"), (5, "backend"),
                           (6, "threshold"), (7, "benign")):
            with self.subTest(menu=item):
                self.runner.run(item)
                directory = self.verify_artifact(name)
                self.assertTrue((directory / "summary.json").is_file())
                self.assertTrue((directory / "summary.csv").is_file())
                reports = list(directory.rglob("metrics.json"))
                self.assertTrue(reports)
                for report in reports:
                    self.verify_report(report.parent, None if item == 7 else ids, episode_level=item != 7)
        self.verify_baselines(ids)
        self.verify_channels(rows)
        self.verify_backends(first)
        self.verify_thresholds()
        self.verify_benign_workloads(set(ids))
        archive_path = self.runner.run(8)
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())
            checksums = json.loads(archive.read("checksums.json"))["files"]
            self.assertIn("results/configuration_stage2/manifest.json", checksums)
            self.assertIn("effective_config.json", checksums)
            for name, digest in checksums.items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
            self.assertFalse(any(".staging/" in name or name.endswith((".pth", ".safetensors")) for name in checksums))
        operations = [call["operation"] for call in self.executor.calls]
        self.assertEqual(operations[:3], ["train", "train", "score"])
        self.assertIn("baseline", operations)
        self.assertIn("infer", operations)
        self.assertFalse(self.executor.active)

    def verify_baselines(self, ids):
        directory = self.results / "baselines"
        parts = [read_jsonl(directory / name / "prefix_decisions.jsonl")
                 for name in ("sentinel", "prompt_guard", "raguard")]
        combined = read_jsonl(directory / "combined_or/prefix_decisions.jsonl")
        self.assertEqual([r["prediction"] for r in combined],
                         [int(any(part[i]["prediction"] for part in parts)) for i in range(len(ids))])
        self.assertTrue(all(r["prediction"] == 0 for r in read_jsonl(directory / "no_defense/prefix_decisions.jsonl")))
        self.assertEqual(sum(c["operation"] == "baseline" for c in self.executor.calls), 1)
        channels = {call[2] for call in self.backend.calls if call[0] == "score_detector"}
        self.assertIn(("prompt",), channels)
        self.assertIn(("tool",), channels)

    def verify_channels(self, rows):
        directory = self.results / "channel_ablation"
        for seed in (42, 43):
            requests = read_jsonl(directory / f"seed_{seed}/l3/requests.jsonl")
            conditions = {r["condition"] for r in requests}
            self.assertEqual(conditions, {"prompt", "retrieval", "tool", "history"})
            self.assertEqual(len(requests), len(rows) * 4)
            for req in requests:
                source = next(r for r in rows if r["id"] == req["prefix_id"])
                channel = None if req["condition"] == "history" else req["condition"]
                self.assertEqual(req["payload"], build_payload(source["sequence"], self.config["models"]["l3"]["payload"], channel, "original_prompt"))
                self.assertIn(source["channels"]["prompt"], req["payload"])
            cases = read_jsonl(directory / f"seed_{seed}/correlated_cases.jsonl")
            self.assertTrue(cases)
            self.assertTrue(all(c["episode_id"].startswith("correlated_") for c in cases))

    def verify_backends(self, first):
        for seed in (42, 43):
            expected = {r["prefix_id"] for r in read_jsonl(first / f"seed_{seed}/l3_requests.jsonl")
                        if r["configuration"] == "L1-L2-L3"}
            identities = []
            for candidate in self.config["experiments"]["backend"]["candidates"]:
                directory = self.results / f"backend/seed_{seed}/{candidate['name']}"
                identities.append(read_json(directory / "routing_identity.json")["pending_set_sha256"])
                requests = read_jsonl(directory / "l3/requests.jsonl")
                self.assertEqual({r["prefix_id"] for r in requests}, expected)
            self.assertEqual(len(set(identities)), 1)

    def verify_thresholds(self):
        for path in (self.results / "threshold").rglob("operating_point.json"):
            report = read_json(path)
            self.assertEqual(report["selection_source"], "validation_only")
            self.assertTrue(report["other_models_and_policies_fixed"])
            self.assertEqual(report["l1"]["objective"], "max_validation_f1_subject_to_recall_floor")

    def verify_benign_workloads(self, allowed_ids):
        for path in (self.results / "benign").rglob("workload.json"):
            report = read_json(path)
            ids = report["sample_ids"]
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
            self.assertTrue(set(ids) <= allowed_ids)
            self.assertEqual(report["actual_benign_ratio"], int(5 * report["requested_benign_ratio"]) / 5)
            self.assertEqual(report["profile"], "main")
            self.assertEqual(report["l1_threshold"], 0.5)

    def test_real_config_keeps_l3_prompt_schema_and_payload_assets(self):
        original = load_config(CONFIG)
        self.assertEqual(self.config["models"]["l3"], original["models"]["l3"])
        prompt = self.config["models"]["l3"]["system_prompt"]
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), self.config["models"]["l3"]["system_prompt_sha256"])
        self.assertEqual(self.config["paths"]["workspace"], str(self.workspace))
        self.assertEqual(self.config["runtime"]["device"], "cpu")

    def test_inference_only_settings_do_not_invalidate_dataset_or_training(self):
        self.prepare(training=True)
        before = self.runner.compatible('training', ('dataset',))
        self.config['runtime']['l3_python'] = '/different/vllm/python'
        self.config['experiments']['backend']['candidates'].append({
            'name': 'another_backend', 'backend': 'openai-compatible', 'model': 'remote-model'})
        self.assertEqual(self.runner.compatible('training', ('dataset',)), before)

    def test_missing_prerequisites_do_not_start_workers_or_publish(self):
        for item in range(1, 8):
            with self.subTest(menu=item), self.assertRaises((ValueError, FileNotFoundError)):
                self.runner.run(item)
        self.assertFalse(self.executor.calls)
        self.assertFalse(self.results.exists())
        self.prepare()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.runner.run(2)
        self.assertFalse(self.executor.calls)
        self.assertFalse((self.results / "configuration_stage1").exists())

    def test_overwrite_defaults_to_cancel_and_preserves_completed_artifacts(self):
        self.prepare(training=True)
        before = {str(p.relative_to(self.results)): file_sha256(p) for p in self.results.rglob("*") if p.is_file()}
        calls = len(self.executor.calls)
        for item in (0, 1):
            with self.subTest(menu=item), self.assertRaises(ExistingResultsError):
                self.runner.run(item)
        after = {str(p.relative_to(self.results)): file_sha256(p) for p in self.results.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(len(self.executor.calls), calls)

    def test_stale_config_is_rejected_before_worker_execution(self):
        self.prepare(training=True)
        changed = deepcopy(self.config)
        changed["models"]["encoder"]["revision"] = "different-scientific-revision"
        runner = ResearchRunner(changed, executor=self.executor)
        calls = len(self.executor.calls)
        with self.assertRaises(IncompatibleResultsError):
            runner.run(2)
        self.assertEqual(len(self.executor.calls), calls)
        self.assertFalse((self.results / "configuration_stage1").exists())

    def test_cli_overwrite_cancellation_preserves_real_results_and_never_restarts_workers(self):
        from research_runner.cli import run_selection
        self.prepare(training=True)
        before = {str(p.relative_to(self.results)): file_sha256(p) for p in self.results.rglob("*") if p.is_file()}
        calls = len(self.executor.calls)
        for answer in ("", "n", "no"):
            messages = []
            input_fn = unittest.mock.Mock(return_value=answer)
            self.assertIsNone(run_selection(self.runner, "1", input_fn=input_fn, print_fn=messages.append))
            input_fn.assert_called_once()
            self.assertIn("[y/N]", input_fn.call_args.args[0])
            self.assertTrue(any("Cancelled" in message for message in messages))
        self.assertEqual(len(self.executor.calls), calls)
        self.assertEqual(before, {str(p.relative_to(self.results)): file_sha256(p) for p in self.results.rglob("*") if p.is_file()})

    def test_resume_completed_configuration_is_read_only_and_does_not_rerun_workers(self):
        self.prepare(full=True)
        self.config["experiments"]["configuration"]["resume"] = True
        before = {str(p.relative_to(self.results)): file_sha256(p) for p in self.results.rglob("*") if p.is_file()}
        calls = len(self.executor.calls)
        self.runner.run(2)
        self.assertEqual(len(self.executor.calls), calls)
        self.assertEqual(before, {str(p.relative_to(self.results)): file_sha256(p) for p in self.results.rglob("*") if p.is_file()})

    def test_resume_rejects_artifact_with_wrong_identity(self):
        self.prepare(training=True)
        self.runner.configuration_stage1()
        path = self.results / "configuration_stage1/manifest.json"
        manifest = read_json(path)
        manifest["name"] = "configuration_stage2"
        write_json(path, manifest)
        self.config["experiments"]["configuration"]["resume"] = True
        calls = len(self.executor.calls)
        with self.assertRaises(ArtifactError):
            self.runner.run(2)
        self.assertEqual(len(self.executor.calls), calls)

    def test_wrong_score_ids_abort_transaction_without_publication(self):
        self.prepare(training=True)
        self.backend.bad_score_ids = True
        with self.assertRaisesRegex(ValueError, "Score IDs"):
            self.runner.configuration_stage1()
        self.assertFalse((self.results / "configuration_stage1").exists())
        self.assertFalse(list((self.results / ".staging").iterdir()))

    def test_wrong_response_ids_abort_stage_two_but_preserve_stage_one(self):
        self.prepare(training=True)
        self.runner.configuration_stage1()
        before = file_sha256(self.results / "configuration_stage1/manifest.json")
        self.backend.bad_response_ids = True
        with self.assertRaisesRegex(ValueError, "L3 responses"):
            self.runner.configuration_stage2()
        self.assertEqual(file_sha256(self.results / "configuration_stage1/manifest.json"), before)
        self.verify_artifact("configuration_stage1")
        self.assertFalse((self.results / "configuration_stage2").exists())

    def test_interrupted_stage_two_can_resume_without_rescoring(self):
        self.prepare(training=True)
        self.runner.configuration_stage1()
        self.backend.fail_next_inference = True
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            self.runner.configuration_stage2()
        self.assertFalse((self.results / "configuration_stage2").exists())
        before = sum(call["operation"] == "score" for call in self.executor.calls)
        self.config["experiments"]["configuration"]["resume"] = True
        self.runner.run(2)
        self.assertEqual(sum(call["operation"] == "score" for call in self.executor.calls), before)
        self.verify_artifact("configuration_stage2")


if __name__ == "__main__":
    unittest.main()
