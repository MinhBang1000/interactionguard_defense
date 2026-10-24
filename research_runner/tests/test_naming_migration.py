"""CPU migration and naming contracts; no legacy data or models are mutated."""
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import yaml

from research_runner.artifacts import ArtifactStore, read_json, write_json, write_jsonl
from research_runner.config import ConfigError, load_config
from research_runner.engine import ResearchRunner
from research_runner.payloads import request
from research_runner.scoring import route
from research_runner.naming import paper_order
from tools.migrate_runner_naming import (
    NAMING, PARENTS, label, migrate_artifacts, rename_file, sha, transformed, verify_pair,
)

ROOT = Path(__file__).resolve().parents[2]


class NamingTests(unittest.TestCase):
    def test_mapping_and_rejection_of_old_runtime_orders(self):
        config = load_config(ROOT / 'config/experiments.yaml')
        self.assertEqual(config['naming'], NAMING)
        self.assertEqual(paper_order('Screen-Gate-Audit', config), 'L1-L2-L3')
        self.assertEqual(paper_order('Gate-Screen-Audit', config), 'L2-L1-L3')
        self.assertEqual(label('L2-L1-L3'), 'Screen-Gate-Audit')
        self.assertEqual(label('pending_l3'), 'pending_audit')
        self.assertEqual(set(config['models']), {'encoder', 'autoencoder', 'screen', 'audit'})
        with self.assertRaises(ValueError):
            route([], {}, {}, ['L1'], .5, .5)
        for key in ('gate_recall_floor', 'screen_fpr_budget'):
            self.assertIn(key, config['training']['calibration'])

    def test_opaque_payloads_identifiers_and_probability_values(self):
        original = {'configuration': 'L2-L1-L3', 'l1_threshold': .123,
                    'payload': 'Literal L1-L2-L3 must not change',
                    'sequence': '[PROMPT]\nDiscuss L1', 'id': 'L1_original',
                    'request_id': 'L2_original', 'system_prompt': 'Immutable L3 policy',
                    'layer_scores': {'L2': {'score': .999}},
                    'route': [{'layer': 'L1', 'action': 'Check'}]}
        new = transformed(original)
        self.assertEqual(new['configuration'], 'Screen-Gate-Audit')
        self.assertEqual(new['gate_threshold'], .123)
        self.assertEqual(new['layer_scores'], {'Screen': {'score': .999}})
        self.assertEqual(new['route'], [{'layer': 'Gate', 'action': 'Check'}])
        for key in ('payload', 'sequence', 'id', 'request_id', 'system_prompt'):
            self.assertEqual(new[key], original[key])
        self.assertEqual(original['configuration'], 'L2-L1-L3')

    def test_tokenizer_bytes_and_weight_bytes_are_opaque(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'old/encoder/tokenizer.json'
            write_json(p, {'vocab': {'L1': 1, 'L2': 2}})
            output = root / 'new/encoder/tokenizer.json'
            rename_file(p, output)
            self.assertEqual(p.read_bytes(), output.read_bytes())
            self.assertEqual(verify_pair(p, output), 0)

    def test_every_order_route_uses_functional_names(self):
        config = load_config(ROOT / 'config/experiments.yaml')
        row = {'id': 'x', 'original_trace_id': 'e', 'step_number': 1, 'label': 1}
        for order in config['experiments']['configuration']['orders']:
            output = route([row], {'x': {'score': 1}}, {'x': {'score': 0}},
                           order.split('-'), .5, .5, {'x': 1})[0]
            self.assertTrue({x['layer'] for x in output['route']} <= {'Screen', 'Gate', 'Audit'})
            self.assertIn('pending_audit', output)
            self.assertNotIn('pending_l3', output)

    def test_invalid_paper_mapping_is_not_silently_accepted(self):
        config = yaml.safe_load((ROOT / 'config/experiments.yaml').read_text())
        config['naming']['paper']['Screen'] = 'L2'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.yaml'
            path.write_text(yaml.safe_dump(config))
            with self.assertRaisesRegex(ConfigError, 'naming must use version 2'):
                load_config(path)

    def test_runtime_modules_do_not_import_legacy_implementations(self):
        import ast
        forbidden = ('data', 'eval', 'training', 'models', 'experiments', 'utils', 'config.config')
        for path in (ROOT / 'research_runner').rglob('*.py'):
            if 'tests' in path.parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    self.assertFalse(any(module == prefix or module.startswith(prefix + '.')
                                         for prefix in forbidden), f'{path}:{node.lineno}: {module}')

    def test_copied_payload_rules_have_identical_semantic_bodies(self):
        import ast
        after = ast.parse((ROOT / 'research_runner/implementation/data/data_preprocess.py').read_text())
        functions = {'normalize_retrieval', 'tool_name', 'normalize_trace',
                     'decoded_tool_content', 'audit_trace', 'channel_segment', 'expand_trace'}
        def bodies(tree):
            return {node.name: ast.dump(node, include_attributes=False)
                    for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in functions}
        origins = read_json(ROOT / 'research_runner/implementation/origins.json')
        actual = hashlib.sha256(json.dumps(bodies(after), sort_keys=True).encode()).hexdigest()
        self.assertEqual(actual, origins['payload_rules_semantic_sha256'])
        self.assertEqual(set(bodies(after)), functions)


class ArtifactMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.backup = self.root / 'backup/pre_migration'
        config = yaml.safe_load((ROOT / 'config/experiments.yaml').read_text())
        config['paths']['workspace'] = '..'
        config['runtime']['device'] = 'cpu'
        config['training']['seeds'] = [42]
        p = self.root / 'config/experiments.yaml'
        p.parent.mkdir()
        p.write_text(yaml.safe_dump(config))
        self.config = load_config(p)
        self.target = Path(self.config['paths']['results'])
        write_json(self.root / 'data/splits/split_manifest.json',
                   {'episode_ids': {'train': [], 'val': [], 'test': []}})
        entries = {}
        store = ArtifactStore(self.target, self.config)
        with redirect_stdout(io.StringIO()):
            for name, _ in PARENTS:
                with store.transaction(name, 'old-signature-' + name) as out:
                    write_json(out / 'stage_complete.json', {'stage': 2, 'stage1_manifest_sha256': 'legacy-hash'})
                    row = {'id': 'id_L1', 'configuration': 'L2-L1-L3',
                           'pending_l3': False, 'prediction': 1, 'label': 1,
                           'layer_scores': {'L2': {'score': .9}, 'L1': {'score': .4}},
                           'payload': 'Keep L1 and L3 literally', 'seconds': .025}
                    write_jsonl(out / 'seed_42/L2-L1-L3/prefix_decisions.jsonl', [row])
                    (out / 'weights.safetensors').write_bytes(b'unchanged L1 L2 binary weights')
        import shutil
        shutil.copytree(self.target, self.backup / 'results/research_runner')
        for p in self.target.rglob('*'):
            if p.is_file():
                entries[str(p.relative_to(self.root))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
        write_json(self.backup / 'before.json', {'artifacts': entries, 'git_head': 'fixture'})

    def execute(self, **kwargs):
        with redirect_stdout(io.StringIO()):
            return migrate_artifacts(self.root, self.backup, **kwargs)

    def test_dry_run_does_not_mutate_files(self):
        before = {p: sha(p) for p in self.target.rglob('*') if p.is_file()}
        self.assertEqual(self.execute()['status'], 'dry_run')
        self.assertEqual(before, {p: sha(p) for p in self.target.rglob('*') if p.is_file()})

    def test_migration_preserves_values_ids_and_binary_and_is_idempotent(self):
        result = self.execute(dry_run=False)
        self.assertEqual(result['status'], 'migrated')
        self.assertTrue(result['verification']['binary_bytes_preserved'])
        for name, _ in PARENTS:
            p = self.target / name / 'seed_42/Screen-Gate-Audit/prefix_decisions.jsonl'
            row = json.loads(p.read_text())
            self.assertEqual(row['id'], 'id_L1')
            self.assertEqual(row['configuration'], 'Screen-Gate-Audit')
            self.assertEqual(row['layer_scores']['Screen']['score'], .9)
            self.assertEqual(row['payload'], 'Keep L1 and L3 literally')
            self.assertEqual((self.target / name / 'weights.safetensors').read_bytes(), b'unchanged L1 L2 binary weights')
        self.assertEqual(self.execute(dry_run=False)['status'], 'already_version_2')

    def test_publication_failure_preserves_old_complete_results(self):
        before = {p: sha(p) for p in self.target.rglob('*') if p.is_file()}
        with self.assertRaisesRegex(RuntimeError, 'Injected publication'):
            self.execute(dry_run=False, fail_before_publish=True)
        self.assertEqual(before, {p: sha(p) for p in self.target.rglob('*') if p.is_file()})
        self.assertFalse(list(self.target.parent.glob('.naming-stage-*')))

    def test_changed_backup_is_rejected(self):
        path = self.backup / 'results/research_runner/dataset/weights.safetensors'
        path.write_bytes(b'corrupted')
        with self.assertRaisesRegex(ValueError, 'Backup or live artifact changed'):
            self.execute(dry_run=False)


if __name__ == '__main__':
    unittest.main()
