"""Configuration tests need only the standard library and PyYAML."""

import copy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

from research_runner.artifacts import experiment_signature, fingerprint
from research_runner.config import ConfigError, load_config


class ConfigTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "config" / "experiments.yaml"
        self.path.parent.mkdir()
        self.value = {
            "version": 1,
            "paths": {"workspace": "..", "results": "results", "backup": "backup"},
            "experiments": {"configuration": {
                "terminal_policy": "binary_at_own_threshold", "orders": ["Gate-Screen-Audit"]}},
            "models": {"audit": {"api_key_env": "EXAMPLE_API_KEY", "max_tokens": 32,
                              "system_prompt": "Audit this trace.\n", "schema": {"type": "object"}}},
        }

    def load(self, value=None, **kwargs):
        self.path.write_text(yaml.safe_dump(self.value if value is None else value), encoding="utf-8")
        return load_config(self.path, **kwargs)

    def test_paths_anchor_to_workspace_not_cwd(self):
        config = self.load()
        self.assertEqual(config["paths"]["workspace"], str(self.root))
        self.assertEqual(config["paths"]["results"], str(self.root / "results"))
        self.assertEqual(config["data"]["root"], str(self.root / "data"))
        self.assertFalse(config["experiments"]["configuration"]["resume"])
        self.assertEqual(config["models"]["audit"]["system_prompt"], "Audit this trace.\n")
        self.assertEqual(config["models"]["audit"]["schema"], {"type": "object"})

    def test_root_explicitly_overrides_workspace(self):
        self.value["runtime"] = {"audit_python": "tools/python"}
        self.value["models"]["encoder"] = {"checkpoint_path": "models/encoder"}
        config = self.load(root=self.root / "override")
        self.assertEqual(config["paths"]["workspace"], str(self.root / "override"))
        self.assertEqual(config["runtime"]["audit_python"], str(self.root / "override/tools/python"))
        self.assertEqual(config["models"]["encoder"]["checkpoint_path"], str(self.root / "override/models/encoder"))

    def test_interpreter_symlink_retains_virtualenv_identity(self):
        executable = self.root / "venv/bin/python"
        executable.parent.mkdir(parents=True)
        executable.symlink_to(sys.executable)
        self.value["runtime"] = {"audit_python": "venv/bin/python"}
        self.assertEqual(self.load()["runtime"]["audit_python"], str(executable))

    def test_invalid_yaml_and_duplicate_keys(self):
        for text in ("[", "[]", "null", "version: 1\nversion: 1\n", "version: 1\npaths:\n  results: x\n  results: y\n"):
            with self.subTest(text=text):
                self.path.write_text(text, encoding="utf-8")
                with self.assertRaises(ConfigError):
                    load_config(self.path)

    def test_missing_file_is_config_error(self):
        with self.assertRaises(ConfigError):
            load_config(self.path)

    def test_unknown_structural_settings(self):
        for section in (None, "paths", "runtime", "data", "training", "backup", "models", "experiments"):
            with self.subTest(section=section):
                value = copy.deepcopy(self.value)
                target = value if section is None else value.setdefault(section, {})
                target["typo"] = 7
                with self.assertRaises(ConfigError):
                    self.load(value)

    def test_malformed_types_and_ranges(self):
        cases = [
            {"version": True}, {"version": 2}, {"paths": None},
            {"runtime": {"device": "cuda:-1"}},
            {"runtime": {"max_gpu_memory_mib": True}},
            {"training": {"seeds": []}}, {"training": {"seeds": [42, 42]}},
            {"training": {"seeds": [True]}},
            {"training": {"calibration": {"gate_recall_floor": 1.1}}},
            {"data": {"reference_test_sha256": "stale"}},
            {"data": {"judge": {"skip": "false"}}},
            {"models": {"audit": {"max_tokens": "32"}}},
            {"models": {"audit": {"temperature": float("nan")}}},
            {"backup": {"include_checkpoints": "false"}},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                value = copy.deepcopy(self.value)
                value.update(changes)
                with self.assertRaises(ConfigError):
                    self.load(value)

    def test_binary_terminal_policy_and_audit_last(self):
        for settings in ({"terminal_policy": "Forward"},
                         {"terminal_policy": "binary_at_own_threshold", "orders": ["Audit-Gate"]},
                         {"terminal_policy": "binary_at_own_threshold", "orders": ["Gate-Gate"]},
                         {"terminal_policy": "binary_at_own_threshold", "resume": "false"}):
            with self.subTest(settings=settings):
                value = copy.deepcopy(self.value)
                value["experiments"]["configuration"] = settings
                with self.assertRaises(ConfigError):
                    self.load(value)

    def test_device_and_memory_none_have_safe_fallbacks(self):
        self.value["runtime"] = {"device": None, "max_gpu_memory_mib": None}
        config = self.load()
        self.assertEqual(config["runtime"]["device"], "cpu")
        self.assertGreater(config["runtime"]["max_gpu_memory_mib"], 0)

    def test_benign_sweep_orders_and_legacy_single_order(self):
        self.value['experiments']['benign'] = {'orders': ['Gate-Screen-Audit', 'Screen-Gate-Audit']}
        self.assertEqual(self.load()['experiments']['benign']['orders'], ['Gate-Screen-Audit', 'Screen-Gate-Audit'])
        self.value['experiments']['benign'] = {'order': 'Gate-Screen-Audit'}
        settings = self.load()['experiments']['benign']
        self.assertEqual(settings['orders'], ['Gate-Screen-Audit'])
        self.assertNotIn('order', settings)
        for invalid in ({}, {'orders': []}, {'orders': ['Audit-Gate']},
                        {'orders': ['Gate', 'Gate']}, {'order': 'Gate', 'orders': ['Gate']}):
            self.value['experiments']['benign'] = invalid
            with self.subTest(settings=invalid), self.assertRaises(ConfigError):
                self.load()

    def test_secrets_are_env_names_not_interpolated_values(self):
        with patch.dict(os.environ, {"EXAMPLE_API_KEY": "first-secret"}):
            first = self.load()
        with patch.dict(os.environ, {"EXAMPLE_API_KEY": "second-secret"}):
            second = self.load()
        self.assertEqual(first, second)
        self.assertEqual(fingerprint(first), fingerprint(second))
        self.assertNotIn("first-secret", str(first))
        for key, value in (("api_key", "literal-secret"), ("api_key_env", "${SECRET}"),
                           ("api_key_env", "not an env name")):
            config = copy.deepcopy(self.value)
            config["models"]["audit"][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ConfigError):
                self.load(config)

    def test_multi_configuration_orders_and_primary_are_validated(self):
        orders = ['Screen-Gate-Audit', 'Gate-Screen-Audit']
        for name in ('baselines', 'threshold'):
            self.value['experiments'][name] = {'orders': orders}
        config = self.load()
        self.assertEqual(config['primary_order'], orders[0])
        for name in ('baselines', 'threshold'):
            self.assertEqual(config['experiments'][name]['orders'], orders)
            for settings in ({}, {'order': orders[0]}, {'orders': []},
                             {'orders': [orders[0], orders[0]]}, {'orders': [orders[1]]},
                             {'orders': ['Screen-Unknown-Audit']},
                             {'orders': orders, 'order': orders[0]}):
                value = copy.deepcopy(self.value)
                value['experiments'][name] = settings
                with self.subTest(name=name, settings=settings), self.assertRaises(ConfigError):
                    self.load(value)
        for primary in ('Audit-Gate', '', None, ['Screen-Gate-Audit']):
            value = copy.deepcopy(self.value)
            value['primary_order'] = primary
            with self.subTest(primary=primary), self.assertRaises(ConfigError):
                self.load(value)

    def test_backend_is_fixed_to_primary_screen_gate_audit_and_safe_unique_candidates(self):
        self.value['experiments']['backend'] = {
            'order': 'Screen-Gate-Audit', 'candidates': [{'name': 'local', 'model': 'fake'}]}
        self.load()
        for changes in ({'order': 'Gate-Screen-Audit'}, {'order': None},
                        {'orders': ['Screen-Gate-Audit']}, {'candidates': []},
                        {'candidates': [{'name': '../escape'}]},
                        {'candidates': [{'name': 'x'}, {'name': 'x'}]},
                        {'candidates': [None]}, {'candidates': [{}]}):
            value = copy.deepcopy(self.value)
            value['experiments']['backend'].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ConfigError):
                self.load(value)
        self.value['primary_order'] = 'Gate-Screen-Audit'
        with self.assertRaisesRegex(ConfigError, 'match primary_order'):
            self.load()

    def test_inline_content_hashes(self):
        model = self.value["models"]["audit"]
        model["system_prompt_sha256"] = hashlib.sha256(model["system_prompt"].encode()).hexdigest()
        model["schema_sha256"] = fingerprint(model["schema"])
        self.load()
        model["schema"]["additionalProperties"] = False
        with self.assertRaisesRegex(ConfigError, "does not match"):
            self.load()

    def test_inline_schema_credential_field_names_are_not_credentials(self):
        schema = {"type": "object", "properties": {"token": {"type": "string"}}}
        self.value["models"]["audit"]["schema"] = schema
        self.value["models"]["audit"]["schema_sha256"] = fingerprint(schema)
        self.assertEqual(self.load()["models"]["audit"]["schema"], schema)

    def test_recursive_alias_and_yaml_object_are_rejected(self):
        for text in ("version: 1\nmodels: &a\n  audit: *a\n", "version: 1\nmodels:\n  audit: 2026-01-01\n"):
            self.path.write_text(text, encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(self.path)

    def test_per_experiment_signatures_ignore_other_experiments_and_controls(self):
        first = self.load()
        second = copy.deepcopy(first)
        second["experiments"]["baselines"] = {"resume": False, "fpr_budget": 0.4}
        second["experiments"]["configuration"].update(stage="stage2", resume=True)
        second["paths"]["results"] = "elsewhere"
        second["backup"]["enabled"] = False
        self.assertEqual(experiment_signature(first, "configuration"),
                         experiment_signature(second, "configuration"))
        second["experiments"]["configuration"]["orders"] = ["Gate"]
        self.assertNotEqual(experiment_signature(first, "configuration"),
                            experiment_signature(second, "configuration"))
        self.assertNotEqual(experiment_signature(first, "configuration", {"input": "a"}),
                            experiment_signature(first, "configuration", {"input": "b"}))

    def test_real_campaign_yaml_loads_without_models(self):
        path = Path(__file__).resolve().parents[2] / "config/experiments.yaml"
        if not path.exists():
            self.skipTest("Parent campaign YAML has not been created")
        config = load_config(path)
        self.assertEqual(config["paths"]["workspace"], str(path.parents[1]))
        self.assertEqual(config["experiments"]["configuration"]["terminal_policy"], "binary_at_own_threshold")

    def test_imports_do_not_import_torch(self):
        result = subprocess.run([sys.executable, "-B", "-c",
                                 "import sys; import research_runner.config; import research_runner.artifacts; "
                                 "assert 'torch' not in sys.modules"],
                                cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
