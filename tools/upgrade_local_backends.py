"""Attest the reviewed, opt-in local-backend extension without rewriting results.

Requires a pre-edit capture from upgrade_multiconfig.py. Defaults to dry-run.
Only the reviewed adapter/executor/config/preflight additions and Item 5 method
may change old production code. Scientific config outside Item 5 must be identical.
"""
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
           'channel_ablation': ('dataset',), 'benign': ('configuration_stage1',),
           'backend': ('configuration_stage1', 'configuration_stage2')}
NEW_MODULES = {'research_runner/modern_audit.py', 'research_runner/backend_setup.py',
               'research_runner/backend_reuse.py', 'research_runner/runtime_environment.py'}

# Remove only these reviewed additions before comparing the entire old AST.
ADDITIONS = [
    'from research_runner.modern_audit import OPTION_KEYS, validate_options',
    'if OPTION_KEYS.intersection(config):\n    validate_options(config)',
    'if cfg.get("vllm_version"):\n    from research_runner.modern_audit import check_version\n    check_version(cfg)',
    'if cfg.get("structured_output_api") == "structured_outputs":\n'
    '    from research_runner.modern_audit import modern_vllm\n    return modern_vllm(self, cfg)',
    'from research_runner.modern_audit import validate_options',
    'try:\n    validate_options(candidate)\nexcept ValueError as exc:\n'
    '    raise ConfigError(f"Invalid backend candidate: {exc}") from exc',
    "for candidate in config['experiments'].get('backend', {}).get('candidates', []):\n"
    "    if 'python' in candidate:\n"
    "        executable = Path(candidate['python']).expanduser()\n"
    "        candidate['python'] = os.path.abspath(executable if executable.is_absolute() else workspace / executable)",
    "if llm and (job.get('backend') or {}).get('python'):\n    python = job['backend']['python']",
    'preflight = None',
    "if isinstance(self.executor, ProcessExecutor):\n    preflight = self.executor.preflight_audit(settings['candidates'])",
    "if preflight is not None:\n    write_json(out / 'runtime_preflight.json', preflight)",
    "if llm and backend.get('structured_output_api') == 'structured_outputs':\n"
    "    from research_runner.runtime_environment import worker_environment\n"
    "    environment = worker_environment(python, backend, environment)",
    'if cfg.get("structured_output_api") == "structured_outputs":\n'
    '    from research_runner.modern_audit import infer_modern_audit\n'
    '    return infer_modern_audit(self, requests, cfg)',
]


def normalized(source, scope='backend'):
    additions = ADDITIONS if scope == 'backend' else [
        'from research_runner.threshold_sweep import select_gate_threshold',
        "if name == 'threshold':\n"
        '    from research_runner.threshold_sweep import validate_settings\n'
        '    try:\n        validate_settings(settings)\n'
        '    except ValueError as exc:\n        raise ConfigError(f\'Invalid Gate sweep: {exc}\') from exc',
    ]
    removable = {ast.dump(ast.parse(text).body[0], include_attributes=False) for text in additions}
    keyword = ast.dump(ast.parse('f(**cfg.get("chat_template_kwargs", {}))').body[0].value.keywords[0])

    class Review(ast.NodeTransformer):
        def visit(self, node):
            if ast.dump(node, include_attributes=False) in removable:
                return None
            return super().visit(node)

        def visit_FunctionDef(self, node):
            # Item 5 alone gains exact-reference reuse. Its preserved upstream
            # dependencies and all other scientific methods must still match.
            mutable = {'preflight_audit', 'backend'} if scope == 'backend' else {'threshold'}
            if node.name in mutable:
                return None
            return self.generic_visit(node)

        def visit_keyword(self, node):
            return None if scope == 'backend' and ast.dump(node) == keyword else self.generic_visit(node)

    return ast.dump(Review().visit(ast.parse(source)), include_attributes=False)


def apply(root, backup, *, dry_run=True, scope='backend'):
    from research_runner.artifacts import file_sha256, read_json, write_json
    from research_runner.config import load_config
    from research_runner.engine import ResearchRunner
    from tools.upgrade_multiconfig import source_hashes

    before = read_json(backup / 'before.json')
    if scope not in {'backend', 'threshold'}:
        raise ValueError('Unsupported audited upgrade scope')
    current = load_config(root / 'config/experiments.yaml')
    def scientific(config):
        value = deepcopy(config)
        for key in ('_config_path', '_config_sha256'):
            value.pop(key, None)
        value['experiments'].pop(scope, None)
        return value
    if scientific(current) != scientific(before['config']):
        raise ValueError(f'Scientific inputs outside {scope} changed; cannot preserve previous results')
    after_sources = source_hashes(root)
    changed = {p for p in set(before['sources']) | set(after_sources)
               if before['sources'].get(p) != after_sources.get(p)}
    for relative, digest in before['sources'].items():
        if file_sha256(backup / relative) != digest:
            raise ValueError(f'Pre-edit source backup changed: {relative}')
    for relative in changed:
        modules = NEW_MODULES if scope == 'backend' else {'research_runner/threshold_sweep.py', 'research_runner/threshold_reuse.py'}
        if relative.startswith('research_runner/tests/') or relative in modules:
            continue
        allowed = ({'research_runner/adapters.py', 'research_runner/config.py',
                    'research_runner/execution.py', 'research_runner/engine.py'} if scope == 'backend'
                   else {'research_runner/config.py', 'research_runner/engine.py'})
        if relative not in allowed:
            raise ValueError(f'Unreviewed source change: {relative}')
        if normalized((root / relative).read_text(), scope) != normalized((backup / relative).read_text(), scope):
            raise ValueError(f'Existing scientific behavior changed: {relative}')

    # The original source tree lives in the capture. Absolute data/runtime paths
    # remain the originals; only the directory used for SOURCE hashing changes.
    frozen = deepcopy(before['config'])
    frozen['paths']['workspace'] = str(backup)
    old, new = ResearchRunner(frozen), ResearchRunner(current)
    paths = {name: new.store.load(name) for name in PARENTS if (new.store.root / name).exists()}
    old.prerequisite = new.prerequisite = lambda name: paths[name]
    records, receipts = {}, {}
    for name, path in paths.items():
        manifest = read_json(path / 'manifest.json')
        manifest_hash = file_sha256(path / 'manifest.json')
        original_signature = old.signature(name, PARENTS[name])
        if manifest['signature'] != original_signature:
            previous_path = backup / 'previous_compatibility' / f'{name}.json'
            previous = read_json(previous_path)
            if any(previous.get(k) != v for k, v in {
                    'version': 1, 'name': name, 'parents': list(PARENTS[name]),
                    'manifest_sha256': manifest_hash, 'from_signature': manifest['signature'],
                    'to_signature': original_signature}.items()):
                raise ValueError(f'Pre-upgrade artifact is stale: {name}')
        if name in before['artifacts'] and before['artifacts'][name]['manifest_sha256'] != manifest_hash:
            raise ValueError(f'Completed artifact changed since capture: {name}')
        records[name] = {'manifest_sha256': manifest_hash, 'validated_before_signature': original_signature}
        receipts[name] = {'version': 1, 'name': name, 'parents': list(PARENTS[name]),
                          'manifest_sha256': manifest_hash, 'from_signature': manifest['signature'],
                          'to_signature': new.signature(name, PARENTS[name]), 'backup': str(backup),
                          'reason': ('Reviewed Item 6 validation-only sweep selection/grid; all other scientific settings/results unchanged.'
                                     if scope == 'threshold' else
                                     'Reviewed Item 5 local backends and exact Item 2 reference reuse; existing models/routing/metrics unchanged.'),
                          'source_changes': {p: {'before': before['sources'].get(p), 'after': after_sources.get(p)}
                                             for p in sorted(changed)}}
    if source_hashes(root) != after_sources:
        raise ValueError('Sources changed during verification')
    report = {'status': 'dry_run' if dry_run else 'applied', 'scope': scope, 'artifacts_unchanged': records,
              'source_changes': sorted(changed), 'receipts': receipts,
              'new_config_sha256': file_sha256(root / 'config/experiments.yaml'),
              'backend_configuration': current['experiments']['backend']}
    if scope == 'threshold':
        report['threshold_configuration'] = current['experiments']['threshold']
    if dry_run:
        return report
    directory = new.store.root / '.compatibility'
    directory.mkdir(exist_ok=True)
    prior = {name: (directory / f'{name}.json').read_bytes() if (directory / f'{name}.json').exists() else None
             for name in receipts}
    try:
        for name, receipt in receipts.items():
            write_json(directory / f'{name}.json', receipt)
        if source_hashes(root) != after_sources:
            raise ValueError('Sources changed while sealing receipts')
        for name, path in paths.items():
            if file_sha256(path / 'manifest.json') != records[name]['manifest_sha256']:
                raise ValueError('Artifact changed while sealing receipts')
        write_json(backup / ('threshold_sweep_upgrade.json' if scope == 'threshold' else 'local_backend_upgrade.json'), report)
    except BaseException:
        for name, data in prior.items():
            target = directory / f'{name}.json'
            if data is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(data)
        raise
    return report


def main():
    sys.path.insert(0, str(ROOT))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--scope', choices=['backend', 'threshold'], default='backend')
    args = parser.parse_args()
    import json
    print(json.dumps(apply(ROOT, args.backup.resolve(), dry_run=not args.apply, scope=args.scope), indent=2))


if __name__ == '__main__':
    main()
