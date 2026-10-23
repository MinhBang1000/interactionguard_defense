"""Production adapter contracts, exercised with CPU mocks only."""
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from research_runner import adapters as a


def l3_config():
    from research_runner.config import load_config
    config = copy.deepcopy(load_config(a.ROOT / "config/experiments.yaml")["models"]["l3"])
    if "system_prompt" in config:
        config["prompt"] = config.pop("system_prompt")
    return config


def verdict(**kwargs):
    return {**dict.fromkeys(a.BOOL_KEYS, False), "user_intent": "Summarize", "confidence": .1,
            "evidence_span": "", "reason": "No attack", **kwargs}


def rows(*sequences):
    return [{"id": str(i), "sequence": sequence} for i, sequence in enumerate(sequences)]


class FakeScorer:
    def __init__(self, values):
        self.values = values
        self.cache = {}

    def score(self, texts):
        for text in texts:
            self.cache[text] = self.values[text]


class ImportAndPayloadTests(unittest.TestCase):
    def test_parent_import_and_payload_do_not_import_ml(self):
        code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'transformers', 'vllm', 'numpy', 'openai', 'dotenv', 'huggingface_hub'}:
        raise AssertionError('Heavy parent import: ' + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from research_runner.adapters import RealBackend, build_payload
backend = RealBackend({'runtime': {'device': 'cpu'}})
assert '[PROMPT]\\nhi' in build_payload('[PROMPT]\\nhi')
backend.close()
"""
        subprocess.run([sys.executable, "-c", code], cwd=a.ROOT, check=True, capture_output=True, text=True)

    def test_payload_matches_original_o2_without_importing_its_dependencies(self):
        import ast
        source = a.ROOT / "models/detectors/layer3_detector_o2.py"
        tree = ast.parse(source.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"_trim_middle", "_parse_channels"}]
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Layer3DetectorO2")
        nodes.append(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_build_trace_payload"))
        namespace = {"_SECTION_RE": a.SECTION_RE}
        exec(compile("from __future__ import annotations\n" + "\n".join(ast.unparse(node) for node in nodes), str(source), "exec"), namespace)
        cfg = {"max_prompt_chars": 1800, "max_block_chars": 1200, "max_blocks_per_channel": 2}
        instance = SimpleNamespace(**cfg)
        sequence = "[PROMPT]\n" + "a" * 2300 + "\n[RETRIEVE]\n" + "b" * 1500 + "\n[RETRIEVE]\nsecond\n[RETRIEVE]\nomit\n[TOOL:x]\ntool\n[AGENT]\nagent\n[EOS]\n"
        for text in ("", "unmarked text", sequence, "[PROMPT]\nx\n[SEP]\n[TOOLS]\n{}"):
            self.assertEqual(a.build_payload(text, cfg), namespace["_build_trace_payload"](instance, text))

    def test_device_is_always_worker_local_zero(self):
        for device in ("0", "1", "cuda", "cuda:1"):
            self.assertEqual(a.RealBackend({"runtime": {"device": device}})._device(), "cuda:0")
        self.assertEqual(a.RealBackend({"runtime": {"device": "cpu"}})._device(), "cpu")

    def test_offline_environment_is_scoped(self):
        with patch.dict(os.environ, {"HF_HUB_OFFLINE": "original"}):
            with a._offline():
                self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "original")


class DetectorTests(unittest.TestCase):
    def test_exact_json_mean_then_max_policy_and_details(self):
        scorer = FakeScorer({"question": [.1], "first": [.2, .8], "second": [.6], "danger": [.7, .9]})
        data = rows('[PROMPT]\nquestion\n[RETRIEVE]\n[Doc 1]\n{"text":"first"}\n[Doc 2]\nsecond\n[TOOLS]\n{"result":"danger"}\n[AGENT]\nMUST NOT SCORE')
        with patch.object(a, "_load_window_scorer", return_value=scorer):
            output = a.RealBackend({}).score_detector(data, {"model": "deepset", "threshold": .8})[0]
        self.assertEqual(output["score"], .8)
        self.assertEqual(output["prediction"], 0)
        self.assertEqual(output["decision"], "Forward")
        self.assertEqual([part["channel"] for part in output["channels"]], ["prompt", "retrieval", "retrieval", "tool"])
        self.assertEqual([part["score"] for part in output["channels"]], [.1, .5, .6, .8])
        self.assertEqual(output["channels"][-1]["windows"], [.7, .9])
        self.assertEqual(output["channels"][-1]["payload_index"], 3)
        self.assertNotIn("MUST NOT SCORE", scorer.cache)

    def test_channel_selection_and_strict_stop(self):
        with patch.object(a, "_load_window_scorer", return_value=FakeScorer({"tool": [.9]})):
            output = a.RealBackend({}).score_detector(rows("[PROMPT]\nignore\n[TOOLS]\ntool"), "prompt_guard_2", ["tool"])[0]
        self.assertEqual(output["decision"], "Stop")
        self.assertEqual(output["channels"][0]["payload_index"], 1)
        with self.assertRaises(ValueError):
            a.RealBackend({}).score_detector([], "deepset", ["agent"])

    def test_missing_channels_do_not_load_a_model(self):
        with patch.object(a, "_load_window_scorer") as load:
            output = a.RealBackend({}).score_detector(rows("[AGENT]\nignored"), "deepset")[0]
        load.assert_not_called()
        self.assertEqual(output["score"], 0)
        self.assertEqual(output["channels"], [])

    def test_invalid_or_empty_window_scores_raise(self):
        for values in ([], [float("nan")], [1.1]):
            with self.subTest(values=values), patch.object(a, "_load_window_scorer", return_value=FakeScorer({"x": values})):
                with self.assertRaises(ValueError):
                    a.RealBackend({}).score_detector(rows("x"), "deepset")

    def test_pinning_and_frozen_policy_are_enforced(self):
        for config in ({"model": "unknown"}, {"model": "deepset", "revision": "main"},
                       {"aggregation": "max"}, {"max_length": 256}, {"window_overlap": 32}, {"dtype": "float16"}):
            with self.subTest(config=config), self.assertRaises(ValueError):
                a._classifier_spec(config)
        for key, (repo, revision) in a.MODEL_SPECS.items():
            self.assertEqual(a._classifier_spec(repo)["revision"], revision)
            self.assertEqual(a._classifier_spec(key)["model"], repo)

    def test_switching_models_releases_previous_gpu_owners_and_excludes_load(self):
        clock = [0.]
        references = []
        def load(spec, device):
            clock[0] += 100
            scorer = FakeScorer({"x": [.8]})
            native = scorer.score
            def score(texts):
                clock[0] += 2
                native(texts)
            scorer.score = score
            references.append(weakref.ref(scorer))
            return scorer
        backend = a.RealBackend({})
        with patch.object(a, "_load_window_scorer", side_effect=load), patch.object(a.time, "perf_counter", side_effect=lambda: clock[0]), patch.object(a, "_release_gpu_cache"):
            first = backend.score_detector(rows("x"), "deepset")[0]
            backend.score_detector(rows("x"), "sentinel_v1")
            self.assertEqual(first["seconds"], 2)
            self.assertIsNone(references[0]())
            self.assertEqual(len(backend._scorers), 1)
            backend.close()
            self.assertFalse(backend._scorers)

    def test_label_mapping_and_safe_loader_options(self):
        config = SimpleNamespace(num_labels=2, id2label={0: "LEGIT", 1: "INJECTION"}, pad_token_id=1, model_type="deberta-v2")
        tokenizer = SimpleNamespace(pad_token_id=1)
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        factories = SimpleNamespace(AutoConfig=SimpleNamespace(from_pretrained=Mock(return_value=config)),
                    AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
                    AutoModelForSequenceClassification=SimpleNamespace(from_pretrained=Mock(return_value=model)))
        torch = SimpleNamespace(float32="float32", device=lambda value: value)
        with patch.dict(sys.modules, {"torch": torch, "transformers": factories}), patch.object(a, "_snapshot", return_value="/mock/snapshot"):
            result = a._load_window_scorer(a._classifier_spec("deepset"), "cpu")
            self.assertEqual(result.detector.malicious_index, 1)
            options = factories.AutoModelForSequenceClassification.from_pretrained.call_args.kwargs
            self.assertTrue(options["local_files_only"])
            self.assertTrue(options["use_safetensors"])
            self.assertFalse(options["trust_remote_code"])
            with self.assertRaisesRegex(ValueError, "malicious label"):
                a._load_window_scorer(a._classifier_spec({"model": "deepset", "malicious_label": 0}), "cpu")
            config.id2label = {0: "LABEL_0", 1: "LABEL_1"}
            result = a._load_window_scorer(a._classifier_spec("prompt_guard_2"), "cpu")
            self.assertEqual(result.detector.malicious_index, 1)
            with self.assertRaisesRegex(ValueError, "mapping"):
                a._load_window_scorer(a._classifier_spec("sentinel_v1"), "cpu")


class L1AndTrainingTests(unittest.TestCase):
    def test_l1_strict_threshold_timing_no_legacy_cache_and_one_seed_resident(self):
        backend = a.RealBackend({})
        checkpoint = {"encoder_path": "/mock/encoder", "ae_path": "/mock/ae", "l1_threshold": .5}
        references = []
        clock = [0.]
        def load(meta, device):
            clock[0] += 100
            model = Mock()
            def embeddings(values, batch_size, cache_dir):
                self.assertTrue(Path(cache_dir).is_dir())
                self.assertNotIn("models/layer1", cache_dir)
                clock[0] += 2
                return values
            model.extract_embedding.side_effect = embeddings
            model.compute_reconstruction_errors.return_value = [.5, .6]
            references.append(weakref.ref(model))
            return model
        with patch.object(a, "_load_l1", side_effect=load), patch.object(a.time, "perf_counter", side_effect=lambda: clock[0]), patch.object(a, "_release_gpu_cache"):
            first = backend.score_l1(rows("x", "y"), checkpoint)
            self.assertEqual([item["decision"] for item in first], ["Forward", "Stop"])
            self.assertEqual(first[0]["seconds"], 1.)
            backend.score_l1(rows("x", "y"), {**checkpoint, "ae_path": "/mock/ae2"})
            self.assertEqual(len(backend._l1), 1)
            self.assertIsNone(references[0]())

    def test_l1_hash_mismatch_fails_before_importing_models(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = Path(directory) / "encoder"
            encoder.mkdir()
            (encoder / "training_config.json").write_text('{"trust_remote_code": false}')
            ae = Path(directory) / "ae.pth"
            ae.write_bytes(b"not real weights")
            meta = {"encoder_path": str(encoder), "ae_path": str(ae), "l1_threshold": .5, "encoder_sha256": "wrong"}
            with self.assertRaisesRegex(ValueError, "Encoder hash"):
                a._load_l1(meta, "cpu")
            meta.update(encoder_sha256=a._directory_hash(encoder), ae_sha256="wrong")
            with self.assertRaisesRegex(ValueError, "Autoencoder hash"):
                a._load_l1(meta, "cpu")

    def test_explicit_recipe_ignores_selected_deployment_defaults(self):
        encoder, ae, recall = a._training_recipe({"training": {"calibration": {
            "l1_recall_floor": .999, "ae_selection_recall_floor": .98}}})
        self.assertEqual(encoder["max_length"], 8192)
        self.assertEqual(ae["hidden_dims"], [512, 256, 128])
        self.assertEqual(recall, .98)
        for invalid in ({"pooling": "mean"}, {"precision": "fp32"}, {"unknown": 1}):
            with self.assertRaises(ValueError):
                a._training_recipe({"models": {"encoder": invalid}})

    def test_train_isolated_python_and_published_prepared_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "published/training"
            prepared.mkdir(parents=True)
            (prepared / "manifest.json").write_text("{}")
            config = {"_prepared_root": str(prepared), "runtime": {"device": "cuda:1", "python": sys.executable},
                      "models": {}, "training": {}}
            def run(command, **kwargs):
                request = json.loads(Path(command[-3]).read_text())
                self.assertEqual(request["config"]["data"]["_prepared_root"], str(prepared))
                self.assertEqual(kwargs["env"]["HF_HUB_OFFLINE"], "1")
                self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
                Path(command[-1]).write_text(json.dumps({"encoder_path": "/mock/encoder", "ae_path": "/mock/ae", "l1_threshold": .5}))
            with patch.object(a.subprocess, "run", side_effect=run):
                result = a.RealBackend(config).train(42, root / "model")
            self.assertEqual(result["l1_threshold"], .5)

    def test_publication_rebases_only_output_prefix(self):
        result = a._publish_paths({"encoder_path": "/stage/model/encoder", "external": "/stage/model2/encoder", "seed": 42}, Path("/stage/model"), Path("/final/model"))
        self.assertEqual(result["encoder_path"], "/final/model/encoder")
        self.assertEqual(result["external"], "/stage/model2/encoder")


class L3Tests(unittest.TestCase):
    def test_inline_config_never_reads_historical_results(self):
        config = l3_config()
        with patch.object(a, "_read_json", side_effect=AssertionError("Historical file read")), patch.object(Path, "read_text", side_effect=AssertionError("Historical file read")):
            result = a._l3_config(config)
        self.assertEqual(result["prompt"], config["prompt"])
        self.assertEqual(result["schema"], config["schema"])
        for missing in ("prompt", "schema", "max_tokens"):
            invalid = copy.deepcopy(config)
            invalid.pop(missing)
            with self.assertRaises(ValueError):
                a._l3_config(invalid)

    def test_prompt_schema_revision_and_secret_validation(self):
        config = l3_config()
        invalid = [{**config, "prompt": "different"}, {**config, "revision": "main"}, {**config, "api_key": "not-a-real-key"}]
        schema = copy.deepcopy(config["schema"])
        schema["additionalProperties"] = True
        invalid.append({**config, "schema": schema})
        for item in invalid:
            with self.assertRaises(ValueError):
                a._l3_config(item)

    def test_vllm_request_ids_verdict_or_tokens_batching_and_timing(self):
        config = l3_config()
        config["batch_size"] = 1
        clock = [0.]
        tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kwargs: messages[-1]["content"], encode=lambda text, **kwargs: [1, 2])
        def generate(texts, params, **kwargs):
            clock[0] += 3
            completion = SimpleNamespace(text=json.dumps(verdict(tool_attack=True, is_malicious=False, confidence=.5)), finish_reason="stop", token_ids=[1])
            return [SimpleNamespace(outputs=[completion], prompt_token_ids=[1, 2]) for _ in texts]
        engine = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=Mock(side_effect=generate))
        def load(cfg):
            clock[0] += 100
            return engine, object()
        backend = a.RealBackend({"models": {"l3": config}})
        with patch.object(backend, "_vllm", side_effect=load), patch.object(a.time, "perf_counter", side_effect=lambda: clock[0]):
            result = backend.infer_l3([{"request_id": "0:opaque", "payload": "x"}, {"request_id": "1:opaque", "payload": "x"}])
        self.assertEqual([item["request_id"] for item in result], ["0:opaque", "1:opaque"])
        self.assertEqual([item["prediction"] for item in result], [1, 1])
        self.assertEqual(result[0]["seconds"], 3.)
        self.assertEqual(result[0]["token_counts"]["total_tokens"], 3)
        self.assertEqual(engine.generate.call_count, 2)

    def test_invalid_outputs_never_fall_open(self):
        config = a._l3_config(l3_config())
        item = {"request_id": "i:hash", "payload": "x"}
        for text, finish in ((json.dumps(verdict()), "length"), ("{}", "stop"),
                             (json.dumps(verdict(tool_attack="false")), "stop"),
                             (json.dumps(verdict(confidence=float("nan"))), "stop")):
            with self.subTest(text=text), self.assertRaises(ValueError):
                a._l3_result(item, text, finish, config, 1)
        backend = a.RealBackend({"models": {"l3": l3_config()}})
        with patch.object(backend, "_vllm", side_effect=RuntimeError("secret must not escape")):
            result = backend.infer_l3([item])[0]
        self.assertIsNone(result["prediction"])
        self.assertIsNone(result["verdict"])
        self.assertNotIn("secret", json.dumps(result))

    def test_context_overflow_does_not_generate(self):
        config = l3_config()
        config["max_model_len"] = 512
        tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "x", encode=lambda *args, **kwargs: [1])
        engine = SimpleNamespace(get_tokenizer=lambda: tokenizer, generate=Mock())
        backend = a.RealBackend({"models": {"l3": config}})
        with patch.object(backend, "_vllm", return_value=(engine, None)):
            result = backend.infer_l3([{"request_id": "id", "payload": "x"}])[0]
        self.assertIsNone(result["prediction"])
        engine.generate.assert_not_called()

    def test_http_uses_env_bearer_and_strict_schema_without_saving_secret(self):
        config = {**l3_config(), "backend": "openai-compatible", "base_url": "https://example.test/v1",
                  "api_key_env": "ADAPTER_TEST_KEY", "max_retries": 1}
        body = {"choices": [{"message": {"content": json.dumps(verdict())}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3}}
        opener = Mock()
        opener.open.return_value = io.StringIO(json.dumps(body))
        with patch.dict(os.environ, {"ADAPTER_TEST_KEY": "test-secret"}), patch.object(a.urllib_request, "build_opener", return_value=opener):
            result = a.RealBackend({"models": {"l3": config}}).infer_l3([{"request_id": "id", "payload": "text"}])[0]
        request = opener.open.call_args.args[0]
        self.assertEqual(request.get_header("Authorization"), "Bearer test-secret")
        self.assertTrue(json.loads(request.data)["response_format"]["json_schema"]["strict"])
        self.assertNotIn("test-secret", json.dumps(result))
        self.assertEqual(result["token_counts"]["total_tokens"], 5)

    def test_http_redirect_is_disabled(self):
        self.assertIsNone(a._NoRedirect().redirect_request(None, None, 302, None, {}, "https://other.test"))


class RAGTests(unittest.TestCase):
    def test_sequence_conversion_only_uses_prompt_and_retrieval(self):
        item = a._rag_sample({"id": "x", "sequence": "[PROMPT]\nquery\n[RETRIEVE]\n[Doc 1]\nfirst\n[Doc 2]\nsecond\n[TOOLS]\nsecret tool\n[AGENT]\nsecret agent"})
        self.assertEqual(item["prompt"], "query")
        self.assertEqual(item["reasoning_steps"][0]["content"], "[Doc 1]\nfirst\n\n[Doc 2]\nsecond")

    def test_explicit_models_required(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            a.RealBackend({}).score_rag(rows("x"), [], {})

    def test_existing_baseline_wrapper_calibrates_and_reports_errors(self):
        cfg = {"lm_model": "gpt2", "lm_revision": "a" * 40, "use_ts": False}
        class Detector:
            def __init__(self, **kwargs):
                self.thresholds = None
            def score_doc(self, query, doc):
                if doc == "fail":
                    raise RuntimeError("failure")
                return {"pd_signed": 1., "pm": 2.}
            def fit(self, train, benign_label):
                self.thresholds = {"pd_low": 0., "pd_high": 2., "pm_high": 3.}
            def predict_sample(self, item, **kwargs):
                try:
                    self.score_doc("query", item["sequence"])
                    return {"pred": 1, "raguard_doc_results": []}
                except Exception:
                    return {"pred": 0, "raguard_doc_results": [{"reason": "error: failure"}]}
        module = SimpleNamespace(RAGuardDetector=Detector)
        with patch.dict(sys.modules, {"models.baseline.raguard": module}), patch.object(a, "_snapshot", return_value="/mock/gpt2"):
            result = a.RealBackend({}).score_rag(rows("ok", "fail"), [{"label": 0, "sequence": "benign"}], cfg)
        self.assertEqual(result[0]["prediction"], 1)
        self.assertEqual(result[0]["score_kind"], "native_binary_not_probability")
        self.assertIsNone(result[1]["prediction"])
        self.assertIsNotNone(result[1]["error"])


if __name__ == "__main__":
    unittest.main()
