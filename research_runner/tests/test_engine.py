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

from research_runner.implementation.data.data_preprocess import normalize_trace
from research_runner.implementation.data.data_split import SOURCES, SPLITS
from research_runner.tests.fixtures import episode
from research_runner.implementation.evaluation.asr import episode_hash
from research_runner.implementation.evaluation.prompt_success_judge import PROMPT_VERSION
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
        metadata = {"seed": seed, "gate_threshold": 0.5,
                    "encoder_path": str(encoder), "ae_path": str(ae),
                    "selection_path": str(selection)}
        write_json(selection, {"threshold": 0.5, "selection_source": "validation_only"})
        write_json(output / "checkpoint.json", metadata)
        return metadata

    def score_gate(self, rows, checkpoint):
        self.calls.append(("score_gate", checkpoint["seed"], tuple(r["id"] for r in rows)))
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
        # remain below it, so the actual routing invokes Audit rather than going idle.
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

    def infer_audit(self, requests, backend_config=None):
        self.calls.append(("infer_audit", deepcopy(backend_config), deepcopy(requests)))
        if self.fail_next_inference:
            self.fail_next_inference = False
            raise RuntimeError("Synthetic interrupted Audit worker")
        responses = []
        for request in requests:
            prediction = int(any(text in request["payload"] for text in
                                 ("IMPORTANT:", "'get_information' tool")))
            responses.append({"request_id": request["request_id"], "prediction": prediction,
                              "seconds": 0.003, "cache_used": False,
                              "timing_scope": "fake_cpu_uncached_request",
                              "backend": (backend_config or self.config["models"]["audit"])["backend"]})
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
        contexts.enter_context(patch("research_runner.implementation.evaluation.prompt_success_judge.judge_episode", side_effect=AssertionError("No model calls")))
        contexts.enter_context(patch("research_runner.implementation.evaluation.prompt_success_judge.run_vllm", side_effect=AssertionError("No GPU calls")))
        self.config = load_config(CONFIG, root=self.workspace)
        self.config["runtime"].update(device="cpu", python=sys.executable, audit_python=sys.executable)
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

    def snapshot_artifacts(self):
        return {str(path.relative_to(self.results)): file_sha256(path)
                for path in self.results.rglob('*')
                if path.is_file() and '.progress' not in path.relative_to(self.results).parts}

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

    def verify_dataset_metadata(self, directory):
        report = read_json(directory / 'dataset_report.json')
        self.assertEqual(report, read_json(directory / 'dataset_manifest.json'))
        for relative, meta in report['artifacts'].items():
            path = directory / relative
            with self.subTest(dataset_file=relative):
                self.assertEqual(file_sha256(path), meta['sha256'])
                self.assertEqual(path.stat().st_size, meta['bytes'])
                if path.suffix == '.jsonl':
                    self.assertEqual(len(read_jsonl(path)), meta['rows'])
        self.assertEqual(report['output'], str(directory))
        self.assertEqual(read_json(directory / 'splits/split_manifest.json')['source_dir'],
                         str(directory / 'annotated'))
        self.assertEqual(read_json(directory / 'asr/test_baseline_asr_v3.json')['provenance']['traces_path'],
                         str(directory / 'splits/test.jsonl'))
        from research_runner.implementation.data.prepare_training import load_prepared
        for stage in ('encoder', 'autoencoder'):
            for split in ('train', 'val'):
                self.assertTrue(load_prepared(stage, split, output_dir=directory / 'training'))
        for split, source in read_json(directory / 'training/manifest.json')['sources'].items():
            self.assertEqual(source['path'], str(directory / f'processed/{split}_processed.jsonl'))
        for path in directory.rglob('*.json'):
            self.assertNotIn(str(self.results / '.staging') + '/', path.read_text())

    def test_dataset_publication_keeps_nested_hashes_bytes_and_final_paths_consistent(self):
        self.runner.run(0)
        self.verify_dataset_metadata(self.verify_artifact('dataset'))

    def verify_report(self, directory, expected_ids=None, episode_level=True):
        decisions = read_jsonl(directory / "prefix_decisions.jsonl")
        ids = [decision["id"] for decision in decisions]
        self.assertEqual(len(ids), len(set(ids)))
        if expected_ids is not None:
            self.assertEqual(ids, expected_ids)
        self.assertTrue(all(type(d["prediction"]) is int and d["prediction"] in (0, 1) for d in decisions))
        self.assertTrue(all(not d.get("pending_audit", False) for d in decisions))
        stats = read_json(directory / "metrics.json")
        self.assertEqual(stats["samples"], len(ids))
        self.assertEqual(stats["sample_ids"], ids)
        self.assertEqual(sum(stats[key] for key in ("tp", "fp", "fn", "tn")), len(ids))
        timing = read_json(directory / "timing.json")
        self.assertEqual(timing["interpretation"], "component_based_pipeline_estimate_not_direct_end_to_end")
        self.assertGreaterEqual(timing["estimated_pipeline_seconds_per_sample"], 0)
        for layer in ("Gate", "Screen", "Audit"):
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
        self.verify_dataset_metadata(dataset)
        training = self.verify_artifact("training")
        self.assertEqual(read_json(dataset / "splits/split_manifest.json")["episode_ids"], self.members)
        checkpoints = read_json(training / "checkpoints.json")
        self.assertEqual(set(checkpoints), {"42", "43"})
        for checkpoint in checkpoints.values():
            self.assertEqual(checkpoint["gate_threshold"], 0.5)
            self.assertTrue(Path(checkpoint["encoder_path"]).is_dir())
            self.assertTrue(Path(checkpoint["ae_path"]).is_file())
            self.assertNotIn(".staging", checkpoint["encoder_path"])
        self.runner.run(2)
        first = self.verify_artifact("configuration_stage1")
        second = self.verify_artifact("configuration_stage2")
        rows = self.runner.rows()
        ids = [r["id"] for r in rows]
        for seed in (42, 43):
            requests = read_jsonl(first / f"seed_{seed}/audit_requests.jsonl")
            self.assertTrue(requests)
            for order in self.config["experiments"]["configuration"]["orders"]:
                staged = read_jsonl(first / f"seed_{seed}/{order}/routing.jsonl")
                self.assertEqual([d["id"] for d in staged], ids)
                pending = {d["id"] for d in staged if d["pending_audit"]}
                self.assertEqual(pending, {r["prefix_id"] for r in requests if r["configuration"] == order})
                self.verify_report(second / f"seed_{seed}/{order}", ids)
            for request in requests:
                source = next(r for r in rows if r["id"] == request["prefix_id"])
                self.assertEqual(request["payload"], build_payload(source["sequence"], self.config["models"]["audit"]["payload"]))
                self.assertEqual(request["request_id"], hashlib.sha256(request["payload"].encode()).hexdigest())
            timing = read_json(second / f"seed_{seed}/audit/timing.json")
            self.assertTrue(timing["uncached_measured"])
            self.assertFalse(timing["cache_used"])
            self.assertEqual(timing["submitted_requests"], len(requests))
            responses = read_jsonl(second / f"seed_{seed}/audit/aligned_responses.jsonl")
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
        # The real serialization pipeline must emit functional names, not merely
        # accept them in an in-memory routing unit test.
        import re
        old_token = re.compile(r'(?<![A-Za-z0-9])[lL][123](?=$|[^A-Za-z0-9])')
        for path in self.results.rglob('*'):
            self.assertIsNone(old_token.search(path.name), str(path))
        snapshot = read_json(second / 'effective_config.json')
        self.assertEqual(snapshot['naming'], self.config['naming'])
        for row in read_jsonl(second / 'seed_42/Screen-Gate-Audit/prefix_decisions.jsonl'):
            self.assertNotIn('pending_l3', row)
            self.assertTrue(set(row['layer_scores']) <= {'Screen', 'Gate', 'Audit'})
        archive_path = self.runner.run(9)
        for name in ('dataset', 'training', 'configuration', 'baselines', 'channel_ablation',
                     'backend', 'threshold', 'benign', 'backup'):
            progress_root = self.results / '.progress' / name
            status = read_json(progress_root / 'progress.json')
            self.assertEqual(status['status'], 'completed', name)
            self.assertTrue(read_jsonl(progress_root / 'progress.jsonl'))
        with zipfile.ZipFile(archive_path) as archive:
            self.assertIsNone(archive.testzip())
            self.assertFalse(any('.progress' in name.split('/') for name in archive.namelist()))
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
        summary = read_json(directory / 'summary.json')
        for name in ('sentinel', 'prompt_guard', 'raguard', 'combined_or', 'no_defense'):
            self.assertEqual(summary[name]['trials'], 1)
            self.assertIn('not_independent_repeats', summary[name]['variation'])
        for seed in self.config['training']['seeds']:
            for order in self.config['experiments']['baselines']['orders']:
                target = directory / f'seed_{seed}/InteractionGuard/{order}'
                source = self.results / f'configuration_stage2/seed_{seed}/{order}/prefix_decisions.jsonl'
                reused = self.verify_report(target, ids)
                self.assertEqual([{k: v for k, v in d.items() if k not in ('seed', 'configuration')}
                                  for d in reused], read_jsonl(source))
                provenance = read_json(target / 'reuse.json')
                self.assertEqual(provenance['source_sha256'], file_sha256(source))
                self.assertEqual(provenance['new_inference_calls'], 0)
                self.assertEqual(provenance['primary'], order == self.config['primary_order'])
                group = summary[f'InteractionGuard/{order}']
                self.assertEqual(group['trials'], len(self.config['training']['seeds']))
                asr = read_json(target / 'asr.json')
                self.assertIn('macro', group['asr'])
                self.assertEqual(group['asr']['categories']['correlated']['attempts_per_trial'],
                                 [asr['per_attack_type']['correlated']['total']] * group['trials'])
        table = (directory / 'summary.csv').read_text()
        for order in self.config['experiments']['baselines']['orders']:
            self.assertIn(f'InteractionGuard/{order}', table)
        self.assertIn('asr.category.correlated', table)

    def verify_channels(self, rows):
        directory = self.results / "channel_ablation"
        for seed in (42, 43):
            requests = read_jsonl(directory / f"seed_{seed}/audit/requests.jsonl")
            conditions = {r["condition"] for r in requests}
            self.assertEqual(conditions, {"prompt", "retrieval", "tool", "history"})
            self.assertEqual(len(requests), len(rows) * 4)
            for req in requests:
                source = next(r for r in rows if r["id"] == req["prefix_id"])
                channel = None if req["condition"] == "history" else req["condition"]
                self.assertEqual(req["payload"], build_payload(source["sequence"], self.config["models"]["audit"]["payload"], channel, "original_prompt"))
                self.assertIn(source["channels"]["prompt"], req["payload"])
            cases = read_jsonl(directory / f"seed_{seed}/correlated_cases.jsonl")
            self.assertTrue(cases)
            self.assertTrue(all(c["episode_id"].startswith("correlated_") for c in cases))

    def verify_backends(self, first):
        for seed in (42, 43):
            order = self.config['experiments']['backend']['order']
            expected = [r for r in read_jsonl(first / f"seed_{seed}/audit_requests.jsonl")
                        if r["configuration"] == order]
            identities = []
            for candidate in self.config["experiments"]["backend"]["candidates"]:
                directory = self.results / f"backend/seed_{seed}/{candidate['name']}"
                identities.append(read_json(directory / "routing_identity.json")["pending_set_sha256"])
                requests = read_jsonl(directory / "audit/requests.jsonl")
                self.assertEqual(requests, expected)
                identity = read_json(directory / 'routing_identity.json')
                self.assertEqual(identity['configuration'], 'Screen-Gate-Audit')
                self.assertEqual(identity['ordered_prefix_ids'], [r['prefix_id'] for r in expected])
            self.assertEqual(len(set(identities)), 1)

    def verify_thresholds(self):
        paired = {}
        for path in (self.results / "threshold").rglob("operating_point.json"):
            report = read_json(path)
            self.assertEqual(report["selection_source"], "validation_only")
            self.assertTrue(report["other_models_and_policies_fixed"])
            expected = ('max_validation_f1_subject_to_recall_floor' if report['operating_point'] == 'validation_max_f1'
                        else 'max_threshold_under_recall_floor')
            self.assertEqual(report['gate']['objective'], expected)
            self.assertEqual(report['sweep_selection_policy'], 'max_threshold_under_recall_floor')
            key = (report['seed'], report['operating_point'])
            shared = {k: v for k, v in report.items() if k not in ('configuration', 'primary')}
            if key not in paired:
                paired[key] = (shared, set())
            self.assertEqual(paired[key][0], shared)
            if 'configuration' in report:
                order = report['configuration']
                paired[key][1].add(order)
                decisions = read_jsonl(path.parent / 'prefix_decisions.jsonl')
                self.assertEqual([d['id'] for d in decisions], report['sample_ids'])
                self.assertTrue(all(d['route'][0]['layer'] == order.split('-')[0] for d in decisions))
        for shared, orders in paired.values():
            self.assertEqual(orders, set(self.config['experiments']['threshold']['orders']))
        summary = read_json(self.results / 'threshold/summary.json')
        expected = {f'{point}/{order}' for _, point in paired
                    for order in self.config['experiments']['threshold']['orders']}
        self.assertEqual(set(summary), expected)
        self.assertTrue(all('asr' in value for value in summary.values()))

    def test_multi_configuration_reuses_scores_verdicts_and_preserves_prerequisites(self):
        self.prepare(full=True)
        frozen = self.snapshot_artifacts()
        settings = deepcopy(self.config)
        before = len(self.backend.calls)
        self.runner.run(3)
        calls = self.backend.calls[before:]
        self.assertFalse(any(c[0] in ('train', 'score_gate', 'infer_audit') for c in calls))
        self.verify_baselines([r['id'] for r in self.runner.rows()])
        for item in (5, 6):
            before = len(self.backend.calls)
            self.runner.run(item)
            self.assertTrue(all(c[0] == 'infer_audit' for c in self.backend.calls[before:]))
        self.verify_backends(self.results / 'configuration_stage1')
        self.verify_thresholds()
        self.assertEqual(settings, self.config)
        after = self.snapshot_artifacts()
        self.assertTrue(all(after[path] == digest for path, digest in frozen.items()))
        for name, item in (('baselines', 3), ('backend', 5), ('threshold', 6)):
            self.config['experiments'][name]['resume'] = True
            before = len(self.executor.calls)
            self.runner.run(item)
            self.assertEqual(len(self.executor.calls), before)

    def test_missing_baseline_order_is_rejected_before_detector_calls(self):
        self.config['experiments']['configuration']['orders'] = ['Gate-Screen-Audit']
        self.prepare(full=True)
        calls = len(self.executor.calls)
        with self.assertRaisesRegex(ValueError, 'Missing Experiment 2 configuration Screen-Gate-Audit'):
            self.runner.run(3)
        self.assertEqual(calls, len(self.executor.calls))
        self.assertFalse((self.results / 'baselines').exists())

    def test_threshold_reroutes_each_order_with_the_same_scores_and_threshold(self):
        self.prepare(training=True)
        rows = self.runner.rows()
        selected = next(r['id'] for r in rows if r['label'] and r['step_number'] % 2)
        original = self.backend.score_gate
        def scores(samples, checkpoint):
            records = original(samples, checkpoint)
            return [{**r, 'score': .05} if r['id'] == selected else r for r in records]
        with patch.object(self.backend, 'score_gate', side_effect=scores):
            self.runner.configuration_stage1()
        before = len(self.backend.calls)
        self.runner.run(6)
        self.assertTrue(all(c[0] == 'infer_audit' for c in self.backend.calls[before:]))
        for seed in self.config['training']['seeds']:
            point = self.results / f'threshold/seed_{seed}/validation_max_f1'
            lookup = {}
            for order in self.config['experiments']['threshold']['orders']:
                decisions = read_jsonl(point / order / 'prefix_decisions.jsonl')
                lookup[order] = next(r for r in decisions if r['id'] == selected)
                requests = read_jsonl(point / order / 'audit/requests.jsonl')
                self.assertTrue(all(r['seed'] == seed and r['configuration'] == order
                                    and r['operating_point'] == 'validation_max_f1' for r in requests))
            self.assertEqual(lookup['Screen-Gate-Audit']['prediction'], 1)
            self.assertEqual(lookup['Gate-Screen-Audit']['prediction'], 0)
        self.verify_thresholds()

    def test_backend_missing_primary_order_aborts_before_inference(self):
        self.config['experiments']['configuration']['orders'] = ['Gate-Screen-Audit']
        self.prepare(training=True)
        self.runner.configuration_stage1()
        before = len(self.executor.calls)
        with self.assertRaisesRegex(ValueError, 'Missing Experiment 2 configuration Screen-Gate-Audit'):
            self.runner.run(5)
        self.assertEqual(len(self.executor.calls), before)
        self.assertFalse((self.results / 'backend').exists())

    def test_backend_requests_reject_missing_duplicate_reordered_or_wrong_payload(self):
        self.prepare(training=True)
        self.runner.configuration_stage1()
        first = self.results / 'configuration_stage1'
        rows = self.runner.rows()
        source = first / 'seed_42/audit_requests.jsonl'
        original = read_jsonl(source)
        pending = [r for r in original if r['configuration'] == 'Screen-Gate-Audit']
        self.assertGreater(len(pending), 1)
        other = [r for r in original if r['configuration'] != 'Screen-Gate-Audit']
        for altered in (pending[:-1], pending + pending[:1], list(reversed(pending)),
                        [{**pending[0], 'payload': 'different'}, *pending[1:]],
                        [{**pending[0], 'seed': 43}, *pending[1:]]):
            write_jsonl(source, other + altered)
            with self.subTest(altered=altered[0]), self.assertRaisesRegex(ValueError, 'pending requests|mismatch'):
                self.runner._backend_inputs(first, 42, rows, 'Screen-Gate-Audit')
        write_jsonl(source, original)
        path = first / 'seed_42/Screen-Gate-Audit/routing.jsonl'
        decisions = read_jsonl(path)
        for altered in (decisions[:-1], decisions + decisions[:1], list(reversed(decisions)),
                        [{**decisions[0], 'label': 1 - decisions[0]['label']}, *decisions[1:]]):
            write_jsonl(path, altered)
            with self.assertRaisesRegex(ValueError, 'IDs|metadata mismatch'):
                self.runner._backend_inputs(first, 42, rows, 'Screen-Gate-Audit')

    def test_threshold_atomic_failure_preserves_completed_multi_configuration_results(self):
        self.prepare(training=True)
        self.runner.configuration_stage1()
        self.runner.run(6)
        before = self.snapshot_artifacts()
        with self.assertRaises(ExistingResultsError):
            self.runner.run(6)
        self.backend.fail_next_inference = True
        with self.assertRaisesRegex(RuntimeError, 'interrupted'):
            self.runner.run(6, overwrite=True)
        after = self.snapshot_artifacts()
        self.assertTrue(all(after[path] == digest for path, digest in before.items()))
        self.assertTrue(all(path.startswith('.failed/') for path in after.keys() - before.keys()))
        self.verify_artifact('threshold')
        self.assertFalse(list((self.results / '.staging').iterdir()))

    def verify_benign_workloads(self, allowed_ids):
        paired = {}
        for path in (self.results / "benign").rglob("workload.json"):
            report = read_json(path)
            ids = report["sample_ids"]
            self.assertEqual(len(ids), 5)
            self.assertEqual(len(set(ids)), 5)
            self.assertTrue(set(ids) <= allowed_ids)
            self.assertEqual(report["actual_benign_ratio"], int(5 * report["requested_benign_ratio"]) / 5)
            self.assertEqual(report["profile"], "main")
            self.assertEqual(report["gate_threshold"], 0.5)
            relative = path.relative_to(self.results / 'benign').parts
            key = relative[:2]
            identity = (report['sample_ids'], report['paired_workload_sha256'],
                        report['sampling_seed'], report['gate_threshold'], report['screen_threshold'])
            if key not in paired:
                paired[key] = (identity, set())
            self.assertEqual(paired[key][0], identity)
            if 'configuration' in report:
                order = report['configuration']
                paired[key][1].add(order)
                decisions = read_jsonl(path.parent / 'prefix_decisions.jsonl')
                self.assertEqual([d['id'] for d in decisions], ids)
                self.assertTrue(all(d['route'][0]['layer'] == order.split('-')[0] for d in decisions))
                expected = sum(d['label'] == 0 and d['route'][0] == {'layer': 'Gate', 'action': 'Forward'} for d in decisions)
                self.assertEqual(report['benign_front_gate_forward_count'], expected)
        for identity, configurations in paired.values():
            self.assertEqual(configurations, set(self.config['experiments']['benign']['orders']))

    def test_benign_sweep_pairs_orders_without_rescoring_or_retuning(self):
        self.prepare(training=True)
        self.runner.configuration_stage1()
        settings = deepcopy(self.config)
        before = len([c for c in self.executor.calls if c['operation'] in ('train', 'score')])
        self.runner.run(7)
        self.assertEqual(self.config, settings)
        self.assertEqual(before, len([c for c in self.executor.calls if c['operation'] in ('train', 'score')]))
        self.verify_benign_workloads({r['id'] for r in self.runner.rows()})
        path = self.verify_artifact('benign')
        summary = read_json(path / 'summary.json')
        expected = {f'benign_{round(ratio*100)}/{order}'
                    for ratio in settings['experiments']['benign']['ratios']
                    for order in settings['experiments']['benign']['orders']}
        self.assertEqual(set(summary), expected)
        self.assertTrue(all(v['trials'] == len(settings['training']['seeds']) for v in summary.values()))
        self.assertTrue(read_json(path / 'stage_complete.json')['paired_workloads_across_configurations'])

    def test_explicit_upgrade_receipt_keeps_old_artifact_immutable_and_rejects_changes(self):
        self.prepare()
        directory = self.verify_artifact('dataset')
        manifest_path = directory / 'manifest.json'
        manifest = read_json(manifest_path)
        expected = manifest['signature']
        # Simulate a verified artifact from the pre-upgrade runner.
        manifest['signature'] = 'pre-upgrade-signature'
        write_json(manifest_path, manifest)
        frozen = file_sha256(manifest_path)
        receipt = {'version': 1, 'name': 'dataset', 'parents': [],
                   'from_signature': manifest['signature'], 'to_signature': expected,
                   'manifest_sha256': frozen}
        path = self.results / '.compatibility/dataset.json'
        write_json(path, receipt)
        self.assertEqual(self.runner.compatible('dataset'), directory)
        self.assertEqual(file_sha256(manifest_path), frozen)
        changed = deepcopy(self.config)
        changed['data']['reference_test_sha256'] = '0' * 64
        with self.assertRaises(IncompatibleResultsError):
            ResearchRunner(changed, executor=self.executor).compatible('dataset')
        for key in ('from_signature', 'to_signature', 'manifest_sha256', 'name'):
            write_json(path, {**receipt, key: 'wrong'})
            with self.subTest(field=key), self.assertRaises(IncompatibleResultsError):
                self.runner.compatible('dataset')
        write_json(path, receipt)
        (directory / 'processed/test_processed.jsonl').write_text('{}\n')
        with self.assertRaises(ArtifactError):
            self.runner.compatible('dataset')

    def test_real_config_keeps_audit_prompt_schema_and_payload_assets(self):
        original = load_config(CONFIG)
        self.assertEqual(self.config["models"]["audit"], original["models"]["audit"])
        prompt = self.config["models"]["audit"]["system_prompt"]
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(), self.config["models"]["audit"]["system_prompt_sha256"])
        self.assertEqual(self.config["paths"]["workspace"], str(self.workspace))
        self.assertEqual(self.config["runtime"]["device"], "cpu")

    def test_inference_only_settings_do_not_invalidate_dataset_or_training(self):
        self.prepare(training=True)
        before = self.runner.compatible('training', ('dataset',))
        self.config['runtime']['audit_python'] = '/different/vllm/python'
        self.config['experiments']['backend']['candidates'].append({
            'name': 'another_backend', 'backend': 'openai-compatible', 'model': 'remote-model'})
        self.assertEqual(self.runner.compatible('training', ('dataset',)), before)

    def test_missing_prerequisites_do_not_start_workers_or_publish(self):
        for item in range(1, 8):
            with self.subTest(menu=item), self.assertRaises((ValueError, FileNotFoundError)):
                self.runner.run(item)
        self.assertFalse(self.executor.calls)
        self.assertEqual({p.name for p in self.results.iterdir()}, {'.progress'})
        self.prepare()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.runner.run(2)
        self.assertFalse(self.executor.calls)
        self.assertFalse((self.results / "configuration_stage1").exists())

    def test_overwrite_defaults_to_cancel_and_preserves_completed_artifacts(self):
        self.prepare(training=True)
        before = self.snapshot_artifacts()
        calls = len(self.executor.calls)
        for item in (0, 1):
            with self.subTest(menu=item), self.assertRaises(ExistingResultsError):
                self.runner.run(item)
        after = self.snapshot_artifacts()
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
        before = self.snapshot_artifacts()
        calls = len(self.executor.calls)
        for answer in ("", "n", "no"):
            messages = []
            input_fn = unittest.mock.Mock(return_value=answer)
            self.assertIsNone(run_selection(self.runner, "1", input_fn=input_fn, print_fn=messages.append))
            input_fn.assert_called_once()
            self.assertIn("[y/N]", input_fn.call_args.args[0])
            self.assertTrue(any("Cancelled" in message for message in messages))
        self.assertEqual(len(self.executor.calls), calls)
        self.assertEqual(before, self.snapshot_artifacts())
        self.assertEqual(read_json(self.results / '.progress/training/progress.json')['status'], 'cancelled')

    def test_resume_completed_configuration_is_read_only_and_does_not_rerun_workers(self):
        self.prepare(full=True)
        self.config["experiments"]["configuration"]["resume"] = True
        before = self.snapshot_artifacts()
        calls = len(self.executor.calls)
        self.runner.run(2)
        self.assertEqual(len(self.executor.calls), calls)
        self.assertEqual(before, self.snapshot_artifacts())

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
        with self.assertRaisesRegex(ValueError, "Audit responses"):
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
