"""Attest an additive Item 8/menu-9-backup change; never rewrite old results."""
import argparse
import ast
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PARENTS = {'dataset': (), 'training': ('dataset',),
           'configuration_stage1': ('dataset', 'training'),
           'configuration_stage2': ('configuration_stage1',),
           'baselines': ('dataset', 'configuration_stage2'),
           'channel_ablation': ('dataset',), 'backend': ('configuration_stage1', 'configuration_stage2'),
           'threshold': ('configuration_stage1', 'configuration_stage2'), 'benign': ('configuration_stage1',)}


def normalized(source):
    """Remove only the new dispatch/config/worker arm, not any old scientific code."""
    class Review(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            if node.name == 'auditor_targeted':
                return None
            node = self.generic_visit(node)
            if node.name in {'run', 'cancel'}:
                class BackupNumber(ast.NodeTransformer):
                    def visit_Constant(self, value):
                        if value.value == 9:
                            value.value = 8
                        return value
                node = BackupNumber().visit(node)
            return node

        def visit_Assign(self, node):
            targets = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if 'NAMES' in targets and isinstance(node.value, ast.Dict):
                pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values) if k.value != 8]
                node.value.keys, node.value.values = map(list, zip(*pairs))
            if 'MENU' in targets and isinstance(node.value, ast.Dict):
                pairs = [(k, v) for k, v in zip(node.value.keys, node.value.values)
                         if v.value != 'Auditor-Targeted Case Stress Test']
                for key, value in pairs:
                    if key.value == '9' and value.value == 'Backup Snapshot':
                        key.value = '8'
                node.value.keys, node.value.values = map(list, zip(*pairs))
            if '_EXPERIMENTS' in targets and isinstance(node.value, ast.Set):
                node.value.elts = [e for e in node.value.elts if e.value != 'auditor_targeted']
            return self.generic_visit(node)

        def visit_If(self, node):
            test = ast.unparse(node.test)
            if test == "'auditor_targeted' in config['experiments']":
                return None
            if test == "operation == 'auditor_targeted_score'":
                if len(node.orelse) != 1 or not isinstance(node.orelse[0], ast.If):
                    raise ValueError('Unexpected worker branch shape')
                return self.visit(node.orelse[0])
            return self.generic_visit(node)

        def visit_Constant(self, node):
            if node.value == 'Invalid selection. Choose 0-9 or Q.':
                node.value = 'Invalid selection. Choose 0-8 or Q.'
            return node

    return ast.dump(Review().visit(ast.parse(source)), include_attributes=False)


def apply(root, backup, *, dry_run=True):
    from research_runner.artifacts import file_sha256, read_json, write_json
    from research_runner.config import load_config
    from research_runner.engine import ResearchRunner
    from tools.upgrade_multiconfig import source_hashes
    before = read_json(backup / 'before.json')
    config = load_config(root / 'config/experiments.yaml')

    def science(cfg):
        value = deepcopy(cfg)
        value.pop('_config_path', None)
        value.pop('_config_sha256', None)
        value['experiments'].pop('auditor_targeted', None)
        return value

    if science(config) != science(before['config']):
        raise ValueError('Existing experiment settings changed; cannot preserve compatibility')
    sources = source_hashes(root)
    changed = {p for p in set(sources) | set(before['sources']) if sources.get(p) != before['sources'].get(p)}
    allowed = {'research_runner/engine.py', 'research_runner/config.py',
               'research_runner/cli.py', 'research_runner/worker.py'}
    for relative, digest in before['sources'].items():
        if file_sha256(backup / relative) != digest:
            raise ValueError('Pre-edit source backup changed')
    for relative in changed:
        if relative.startswith('research_runner/tests/') or relative == 'research_runner/auditor_targeted.py':
            continue
        if relative not in allowed or normalized((root / relative).read_text()) != normalized((backup / relative).read_text()):
            raise ValueError(f'Existing scientific behavior changed: {relative}')
    old_config = deepcopy(before['config'])
    old_config['paths']['workspace'] = str(backup)
    old, current = ResearchRunner(old_config), ResearchRunner(config)
    paths = {name: current.store.load(name) for name in before['artifacts']}
    old.prerequisite = current.prerequisite = lambda name: paths[name]
    receipts = {}
    for name, path in paths.items():
        manifest = read_json(path / 'manifest.json')
        digest = file_sha256(path / 'manifest.json')
        expected_before = old.signature(name, PARENTS[name])
        if digest != before['artifacts'][name]['manifest_sha256']:
            raise ValueError('Completed manifest changed since capture')
        if manifest['signature'] != expected_before:
            previous = read_json(backup / 'previous_compatibility' / f'{name}.json')
            expected = {'version': 1, 'name': name, 'parents': list(PARENTS[name]),
                        'manifest_sha256': digest, 'from_signature': manifest['signature'], 'to_signature': expected_before}
            if any(previous.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Artifact was already stale before this extension: {name}')
        receipts[name] = {'version': 1, 'name': name, 'parents': list(PARENTS[name]),
                          'manifest_sha256': digest, 'from_signature': manifest['signature'],
                          'to_signature': current.signature(name, PARENTS[name]), 'backup': str(backup),
                          'reason': 'Reviewed additive Item 8 case stress test and Backup moved to 9; old scientific methods/settings/artifacts unchanged.',
                          'source_changes': {p: {'before': before['sources'].get(p), 'after': sources.get(p)} for p in sorted(changed)}}
    if source_hashes(root) != sources:
        raise ValueError('Source changed during verification')
    report = {'status': 'dry_run' if dry_run else 'applied', 'receipts': receipts,
              'artifacts_unchanged': list(paths), 'source_changes': sorted(changed)}
    if dry_run:
        return report
    directory = current.store.root / '.compatibility'
    directory.mkdir(exist_ok=True)
    previous = {name: (directory / f'{name}.json').read_bytes() if (directory / f'{name}.json').exists() else None for name in receipts}
    try:
        for name, receipt in receipts.items():
            write_json(directory / f'{name}.json', receipt)
        if source_hashes(root) != sources:
            raise ValueError('Source changed during sealing')
        for name, path in paths.items():
            if file_sha256(path / 'manifest.json') != receipts[name]['manifest_sha256']:
                raise ValueError('Artifact changed during sealing')
        write_json(backup / 'auditor_targeted_upgrade.json', report)
    except BaseException:
        for name, content in previous.items():
            path = directory / f'{name}.json'
            if content is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(content)
        raise
    return report


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    report = apply(ROOT, args.backup.resolve(), dry_run=not args.apply)
    print({'status': report['status'], 'artifacts_unchanged': report['artifacts_unchanged'], 'source_changes': report['source_changes']})


if __name__ == '__main__':
    main()
