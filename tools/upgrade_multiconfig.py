"""Capture and explicitly attest an audited, code-only multi-config upgrade.

Completed scientific artifacts are never rewritten. Capture must run BEFORE
source/config edits. Apply is opt-in, hash-pinned and rejects scientific drift.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {'dataset': (), 'training': ('dataset',),
           'configuration_stage1': ('dataset', 'training'),
           'configuration_stage2': ('configuration_stage1',),
           'benign': ('configuration_stage1',)}
ALLOWED = {'research_runner/config.py', 'research_runner/engine.py', 'research_runner/plots.py'}


def source_hashes(root):
    from research_runner.artifacts import file_sha256
    return {p.relative_to(root).as_posix(): file_sha256(p)
            for p in (root / 'research_runner').rglob('*.py')}


def preserved_config(config):
    value = deepcopy(config)
    for key in ('_config_path', '_config_sha256', 'primary_order'):
        value.pop(key, None)
    for name in ('baselines', 'backend', 'threshold'):
        value['experiments'].pop(name, None)
    return value


def _unchanged_engine(source):
    """Only the reviewed experiment methods/metadata may change, not inference."""
    tree = ast.parse(source)
    mutable = {'baselines', 'backend', 'threshold', 'aggregate', '_saved_decisions', '_backend_inputs'}
    primary_metadata = ast.dump(ast.parse(
        "if name in {'baselines', 'backend', 'threshold'}:\n"
        "    scientific['primary_order'] = self.config['primary_order']\n").body[0])
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == 'ResearchRunner':
            node.body = [item for item in node.body
                         if not isinstance(item, ast.FunctionDef) or item.name not in mutable]
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == 'signature':
                    item.body = [statement for statement in item.body if ast.dump(statement) != primary_metadata]
    return ast.dump(tree, include_attributes=False)


def verified_artifacts(runner):
    """One full hash pass, then signatures over those verified parent manifests."""
    from research_runner.artifacts import file_sha256, read_json, IncompatibleResultsError
    paths = {name: runner.store.load(name) for name in PARENTS
             if (runner.store.root / name).exists()}
    original = runner.prerequisite
    runner.prerequisite = lambda name: paths[name]
    try:
        manifests = {}
        for name, path in paths.items():
            manifest = read_json(path / 'manifest.json')
            expected = runner.signature(name, PARENTS[name])
            if manifest['signature'] != expected:
                receipt = read_json(runner.store.root / '.compatibility' / f'{name}.json')
                if any(receipt.get(k) != v for k, v in {
                    'version': 1, 'name': name, 'parents': list(PARENTS[name]),
                    'from_signature': manifest['signature'], 'to_signature': expected,
                    'manifest_sha256': file_sha256(path / 'manifest.json')}.items()):
                    raise IncompatibleResultsError(f'Cannot attest stale artifacts: {name}')
            manifests[name] = {'manifest': manifest, 'manifest_sha256': file_sha256(path / 'manifest.json'),
                               'validated_signature': expected}
        return manifests
    finally:
        runner.prerequisite = original


def capture(root, destination):
    from research_runner.artifacts import write_json
    from research_runner.config import load_config
    from research_runner.engine import ResearchRunner
    if destination.exists():
        raise ValueError('Capture destination already exists; never overwrite an upgrade backup')
    config = load_config(root / 'config/experiments.yaml')
    runner = ResearchRunner(config)
    artifacts = verified_artifacts(runner)
    destination.mkdir(parents=True)
    for name in ('research_runner', 'config'):
        shutil.copytree(root / name, destination / name,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(root / 'README.md', destination / 'README.md')
    receipts = runner.store.root / '.compatibility'
    if receipts.exists():
        shutil.copytree(receipts, destination / 'previous_compatibility')
    record = {'version': 1, 'purpose': 'multi_configuration_3_6_fixed_backend_5',
              'captured_at': datetime.now(timezone.utc).isoformat(),
              'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
              'config': config, 'sources': source_hashes(root), 'artifacts': artifacts}
    write_json(destination / 'before.json', record)
    return {'status': 'captured', 'backup': str(destination), 'verified_artifacts': list(artifacts)}


def apply(root, backup, *, dry_run=True):
    from research_runner.artifacts import read_json, write_json, file_sha256
    from research_runner.config import load_config
    from research_runner.engine import ResearchRunner
    before = read_json(backup / 'before.json')
    config = load_config(root / 'config/experiments.yaml')
    if preserved_config(before['config']) != preserved_config(config):
        raise ValueError('Scientific inputs/settings outside Items 3/5/6 changed; do not reuse artifacts')
    current_sources = source_hashes(root)
    changed = {p for p in set(before['sources']) | set(current_sources)
               if before['sources'].get(p) != current_sources.get(p)}
    if any(p not in ALLOWED and not p.startswith('research_runner/tests/') for p in changed):
        raise ValueError(f'Unreviewed production source changes: {sorted(changed - ALLOWED)}')
    if _unchanged_engine((backup / 'research_runner/engine.py').read_text()) != _unchanged_engine(
            (root / 'research_runner/engine.py').read_text()):
        raise ValueError('Scientific orchestration outside the reviewed Item 3/5/6 upgrade changed')
    for relative, digest in before['sources'].items():
        if file_sha256(backup / relative) != digest:
            raise ValueError(f'Backup source snapshot changed: {relative}')
    runner = ResearchRunner(config)
    paths = {}
    for name, record in before['artifacts'].items():
        path = runner.store.load(name)
        if file_sha256(path / 'manifest.json') != record['manifest_sha256']:
            raise ValueError(f'Original completed manifest changed: {name}')
        paths[name] = path
    original = runner.prerequisite
    runner.prerequisite = lambda name: paths[name]
    try:
        receipts = {name: {'version': 1, 'name': name, 'parents': list(PARENTS[name]),
                     'from_signature': record['manifest']['signature'],
                     'to_signature': runner.signature(name, PARENTS[name]),
                     'manifest_sha256': record['manifest_sha256'],
                     'reason': 'Audited Item 3/6 multi-order and Item 5 fixed Screen-Gate-Audit upgrade; scientific artifacts unchanged.',
                     'backup': str(backup), 'source_changes': {
                         p: {'before': before['sources'].get(p), 'after': current_sources.get(p)} for p in sorted(changed)}}
                    for name, record in before['artifacts'].items()}
    finally:
        runner.prerequisite = original
    if dry_run:
        return {'status': 'dry_run', 'artifacts': list(receipts), 'source_changes': sorted(changed)}
    if source_hashes(root) != current_sources:
        raise ValueError('Sources changed during sealing; no receipts written')
    directory = runner.store.root / '.compatibility'
    directory.mkdir(exist_ok=True)
    old = {name: (directory / f'{name}.json').read_bytes()
           if (directory / f'{name}.json').exists() else None for name in receipts}
    try:
        for name, receipt in receipts.items():
            write_json(directory / f'{name}.json', receipt)
        verified_artifacts(runner)
        if source_hashes(root) != current_sources:
            raise ValueError('Sources changed during verification; restoring previous receipts')
        report = {'status': 'applied', 'artifacts_unchanged': list(receipts),
                  'source_changes': sorted(changed), 'receipts': receipts}
        write_json(backup / 'upgrade_record.json', report)
    except BaseException:
        for name, value in old.items():
            path = directory / f'{name}.json'
            if value is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(value)
        raise
    return report


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['capture', 'apply'])
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    result = capture(ROOT, args.backup.resolve()) if args.operation == 'capture' else apply(
        ROOT, args.backup.resolve(), dry_run=not args.apply)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
