"""Code-upgrade receipts bind verified immutable artifacts, not a stale bypass."""
from copy import deepcopy
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

import yaml

from research_runner.artifacts import ArtifactError, IncompatibleResultsError, file_sha256, read_json, write_json
from research_runner.engine import ResearchRunner
from research_runner.tests import test_engine as campaign
from tools import upgrade_multiconfig as upgrade


class UpgradeTests(unittest.TestCase):
    def setUp(self):
        fixture = campaign.EngineTests(methodName='runTest')
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.addCleanup(fixture.tearDown)
        self.fixture = fixture
        self.root = fixture.workspace
        self.runner = fixture.runner
        source = self.root / 'research_runner'
        source.mkdir()
        # Minimal source files are fingerprinted by the real runner in this
        # isolated workspace; execution still uses the production orchestration.
        for name in ('engine.py', 'config.py', 'plots.py'):
            shutil.copy2(campaign.ROOT / 'research_runner' / name, source / name)
        (self.root / 'README.md').write_text('Synthetic upgrade fixture.\n')
        self.yaml = self.root / 'config/experiments.yaml'
        self.original_config = {k: v for k, v in fixture.config.items() if not k.startswith('_')}
        self.save_config(self.original_config)
        fixture.prepare(full=True)
        fixture.runner.run(7)
        self.before = fixture.snapshot_artifacts()
        self.backup = self.root / 'upgrade-backup'
        with patch.object(upgrade.subprocess, 'check_output', return_value='fixture-commit\n'):
            upgrade.capture(self.root, self.backup)
        self.engine = source / 'engine.py'
        self.engine.write_text(self.engine.read_text() + '\n# Synthetic reviewed code-only change.\n')

    def save_config(self, config):
        self.yaml.parent.mkdir(parents=True, exist_ok=True)
        self.yaml.write_text(yaml.safe_dump(config))

    def test_dry_run_then_apply_preserves_all_bytes_and_is_repeatable(self):
        with self.assertRaises(IncompatibleResultsError):
            self.runner.compatible('dataset')
        report = upgrade.apply(self.root, self.backup)
        self.assertEqual(report['status'], 'dry_run')
        self.assertFalse((self.runner.store.root / '.compatibility').exists())
        for _ in range(2):
            report = upgrade.apply(self.root, self.backup, dry_run=False)
            self.assertEqual(set(report['artifacts_unchanged']), set(upgrade.PARENTS))
            self.assertEqual(set(upgrade.verified_artifacts(self.runner)), set(upgrade.PARENTS))
        after = self.fixture.snapshot_artifacts()
        self.assertTrue(all(after[path] == digest for path, digest in self.before.items()))
        self.assertTrue(all(path.startswith('.compatibility/') for path in after.keys() - self.before.keys()))
        self.assertTrue((self.backup / 'upgrade_record.json').exists())
        changed = deepcopy(self.fixture.config)
        changed['models']['audit']['temperature'] = .3
        with self.assertRaises(IncompatibleResultsError):
            ResearchRunner(changed, executor=self.fixture.executor).compatible(
                'configuration_stage2', ('configuration_stage1',))

    def test_scientific_config_drift_cannot_be_sealed(self):
        for section, key, value in (
                ('data', 'reference_test_sha256', '0' * 64),
                ('training', 'seeds', [99]),
                ('models', 'screen', {'model': 'different'}),
                ('models', 'encoder', {'model': 'different'}),
                ('models', 'audit', {'model': 'different'})):
            changed = deepcopy(self.original_config)
            changed[section][key] = value
            self.save_config(changed)
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'Scientific inputs'):
                upgrade.apply(self.root, self.backup, dry_run=False)
        self.save_config(self.original_config)
        self.assertFalse((self.runner.store.root / '.compatibility').exists())

    def test_unreviewed_source_changes_and_corrupted_artifacts_are_rejected(self):
        unexpected = self.root / 'research_runner/detector.py'
        unexpected.write_text('# Must not change a scientific detector.\n')
        with self.assertRaisesRegex(ValueError, 'Unreviewed production source'):
            upgrade.apply(self.root, self.backup)
        unexpected.unlink()
        frozen_source = self.engine.read_text()
        self.engine.write_text(frozen_source.replace("prediction=answer['prediction']", 'prediction=0'))
        with self.assertRaisesRegex(ValueError, 'Scientific orchestration'):
            upgrade.apply(self.root, self.backup)
        self.engine.write_text(frozen_source)
        checkpoint = read_json(self.runner.store.root / 'training/checkpoints.json')['42']
        model = Path(checkpoint['ae_path'])
        original = model.read_bytes()
        model.write_bytes(b'changed weights')
        with self.assertRaises(ArtifactError):
            upgrade.apply(self.root, self.backup, dry_run=False)
        model.write_bytes(original)
        self.assertEqual(self.before[model.relative_to(self.runner.store.root).as_posix()], file_sha256(model))

    def test_failure_rolls_back_receipts_and_keeps_original_results(self):
        old = self.runner.store.root / '.compatibility/dataset.json'
        write_json(old, {'previous': 'receipt retained on failure'})
        frozen = old.read_bytes()
        with patch.object(upgrade, 'verified_artifacts', side_effect=RuntimeError('Synthetic verification failure')):
            with self.assertRaisesRegex(RuntimeError, 'verification failure'):
                upgrade.apply(self.root, self.backup, dry_run=False)
        self.assertEqual(old.read_bytes(), frozen)
        self.assertEqual(list(old.parent.iterdir()), [old])
        self.assertFalse((self.backup / 'upgrade_record.json').exists())
        after = self.fixture.snapshot_artifacts()
        self.assertTrue(all(after[path] == digest for path, digest in self.before.items()))


if __name__ == '__main__':
    unittest.main()
