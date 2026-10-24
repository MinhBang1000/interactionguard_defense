"""Audited, one-way artifact naming migration; never executes model code.

Run with --help after installing the matching v2 source/config revision.
Historical model inputs are opaque. The old source/config is retained in backup.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
LAYERS = {"L1": "Gate", "L2": "Screen", "L3": "Audit"}
NAMING = {"version": 2, "legacy": LAYERS,
          "paper": {"Screen": "L1", "Gate": "L2", "Audit": "L3"}}
OPAQUE = {"system_prompt", "schema", "payload", "sequence", "prompt", "content",
          "text", "raw_text", "response", "completion", "evidence_span", "reason",
          "verdict", "user_intent", "final_answer", "reasoning_steps", "trace",
          "id", "prefix_id", "episode_id", "original_trace_id", "request_id",
          "payload_sha256", "episode_sha256", "sample_ids", "sampled_ids"}
TOKEN = re.compile(r"(?<![A-Za-z0-9])([lL])([123])(?=$|[^A-Za-z0-9])")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def label(text):
    return TOKEN.sub(lambda m: LAYERS['L' + m[2]] if m[1] == 'L'
                     else LAYERS['L' + m[2]].lower(), text)


def transformed(value, key=None):
    """Rename structured metadata, never model inputs, verdict text or IDs."""
    if key in OPAQUE:
        return value
    if isinstance(value, dict):
        result = {label(k): transformed(v, k) for k, v in value.items()}
        if {'version', 'paths', 'models', 'experiments'} <= value.keys():
            result['naming'] = NAMING
        return result
    if isinstance(value, list):
        return [transformed(v, key) for v in value]
    if isinstance(value, str):
        # External model/env/legacy paths are immutable references, not labels.
        if '/' in value and 'results/research_runner/' not in value:
            if value.startswith('/') or value.startswith(('models/', 'experiments/', 'http')):
                return value
        return label(value)
    return value


def numbers(value):
    """Strong numeric/boolean equality independent of naming and provenance."""
    if isinstance(value, dict):
        return {k: numbers(v) for k, v in value.items()
                if k not in {'naming', 'signature', 'files', 'naming_version', 'migration'}
                and not any(s in k.lower() for s in ('sha256', 'checksum', 'fingerprint'))}
    if isinstance(value, list):
        return [numbers(v) for v in value]
    if isinstance(value, (bool, int, float)) or value is None:
        return value
    return value


def rename_file(source, destination):
    """Return whether structured bytes changed; preserve unchanged byte formats."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if 'encoder' in source.parts and source.suffix != '.jsonl':
        # Tokenizer vocabulary and saved HF provenance are model contents, not
        # display labels. Even a token literally named L1 must remain unchanged.
        shutil.copy2(source, destination)
        return False
    if source.suffix == '.jsonl':
        changed = False
        with source.open() as old, destination.open('w') as new:
            for line in old:
                if not line.strip():
                    new.write(line)
                    continue
                value = json.loads(line)
                updated = transformed(value)
                different = updated != value
                changed |= different
                new.write(json.dumps(updated, ensure_ascii=False) + '\n' if different else line)
        return changed
    if source.suffix == '.json':
        value = json.loads(source.read_text())
        updated = transformed(value)
        if updated != value:
            dump(destination, updated)
            return True
    elif source.suffix in {'.csv', '.log'}:
        text = source.read_text()
        updated = label(text)
        if updated != text:
            destination.write_text(updated)
            return True
    shutil.copy2(source, destination)
    return False


def replace_hashes(value, replacements):
    if isinstance(value, dict):
        return {k: replace_hashes(v, replacements) for k, v in value.items()}
    if isinstance(value, list):
        return [replace_hashes(v, replacements) for v in value]
    return replacements.get(value, value) if isinstance(value, str) else value


def verify_pair(before, after):
    if 'encoder' in before.parts and before.suffix != '.jsonl':
        if sha(before) != sha(after):
            raise ValueError(f'Encoder/tokenizer content changed: {before}')
        return 0
    if before.suffix == '.jsonl':
        count = 0
        with before.open() as a, after.open() as b:
            from itertools import zip_longest
            for old, new in zip_longest(a, b):
                if old is None or new is None:
                    raise ValueError(f'Row count changed: {before}')
                if old.strip():
                    x, y = transformed(json.loads(old)), json.loads(new)
                    if numbers(x) != numbers(y):
                        raise ValueError(f'Numeric/boolean content changed: {before}:{count}')
                    for key in OPAQUE:
                        if key in x and x[key] != y.get(key):
                            raise ValueError(f'Protected {key} changed: {before}:{count}')
                    count += 1
        return count
    if before.suffix == '.json':
        if numbers(transformed(json.loads(before.read_text()))) != numbers(json.loads(after.read_text())):
            raise ValueError(f'Numeric/boolean JSON content changed: {before}')
    elif before.suffix == '.csv':
        import csv
        with before.open() as a, after.open() as b:
            expected = [[label(cell) for cell in row] for row in csv.reader(a)]
            if expected != list(csv.reader(b)):
                raise ValueError(f'CSV values changed: {before}')
    elif before.suffix not in {'.log', '.png'} and sha(before) != sha(after):
        raise ValueError(f'Binary/checkpoint changed: {before}')
    return 0


PARENTS = [('dataset', ()), ('training', ('dataset',)),
           ('configuration_stage1', ('dataset', 'training')),
           ('configuration_stage2', ('configuration_stage1',)),
           ('benign', ('configuration_stage1',))]


def migrate_artifacts(root, backup, *, dry_run=True, fail_before_publish=False):
    import sys
    sys.path.insert(0, str(root))
    from research_runner.artifacts import ArtifactStore, fingerprint
    from research_runner.config import load_config
    from research_runner.engine import ResearchRunner
    config = load_config(root / 'config/experiments.yaml')
    if config['naming'] != NAMING:
        raise ValueError('Run and verify source migration first')
    target = Path(config['paths']['results'])
    marker = target / '.naming_migration.json'
    if marker.is_file():
        status = json.loads(marker.read_text())
        if status.get('naming') != NAMING:
            raise ValueError('Unexpected existing migration version')
        runner = ResearchRunner(config)
        for name, parents in PARENTS:
            if (target / name).exists():
                runner.compatible(name, parents)
        return {'status': 'already_version_2', 'record': str(marker)}
    inventory = json.loads((backup / 'before.json').read_text())
    saved = backup / 'results/research_runner'
    entries = inventory['artifacts']
    if dry_run:
        return {'status': 'dry_run', 'files': len(entries),
                'bytes': sum(x['bytes'] for x in entries.values()),
                'backup': str(backup), 'target': str(target), 'naming': NAMING}
    size = sum(x['bytes'] for x in entries.values())
    if shutil.disk_usage(target.parent).free < size * 1.2 + 1024**3:
        raise ValueError('Not enough space for staging and recovery')
    stage = target.parent / ('.naming-stage-' + str(time.time_ns()))
    stage.mkdir()
    record = {'version': 1, 'naming': NAMING, 'backup': str(backup),
              'before_git_head': inventory['git_head'], 'files': {}, 'signatures': {}}
    source_hashes = {str(p.relative_to(root)): sha(p)
                     for p in (root / 'research_runner').rglob('*.py')}
    record['migrated_source_sha256'] = source_hashes
    record['migration_utility_sha256'] = sha(Path(__file__))
    replacements = {}
    try:
        for relative, old in entries.items():
            rel = Path(relative).relative_to('results/research_runner')
            src = saved / rel
            if src.is_symlink() or sha(src) != old['sha256'] or sha(target / rel) != old['sha256']:
                raise ValueError(f'Backup or live artifact changed: {rel}')
            new_rel = Path(label(rel.as_posix()))
            dst = stage / new_rel
            if dst.exists():
                raise ValueError(f'Migration path collision: {new_rel}')
            rename_file(src, dst)
            digest = sha(dst)
            if digest != old['sha256']:
                replacements[old['sha256']] = digest
            record['files'][rel.as_posix()] = {'new_path': new_rel.as_posix(),
                'before_sha256': old['sha256'], 'after_sha256': digest}
            if len(record['files']) % 200 == 0:
                print('STAGED', len(record['files']), flush=True)
        # Workload identity hashes cover renamed threshold keys as well as IDs.
        for path in sorted(stage.glob('benign/seed_*/benign_*/workload.json')):
            value = json.loads(path.read_text())
            old = value.pop('paired_workload_sha256')
            value['paired_workload_sha256'] = fingerprint(value)
            replacements[old] = value['paired_workload_sha256']
            dump(path, value)
        # Propagate file-content changes through nested checksum manifests.
        # This is a DAG; cycles are errors, never silently accepted.
        json_paths = [p for p in stage.rglob('*.json')
                      if p.parent != stage and p.name != 'manifest.json']
        json_paths += [p for p in stage.rglob('manifest.json') if p.parent.parent != stage]
        for iteration in range(20):
            changes = 0
            for path in json_paths:
                value = json.loads(path.read_text())
                updated = replace_hashes(value, replacements)
                if updated != value:
                    before = sha(path)
                    dump(path, updated)
                    after = sha(path)
                    replacements[before] = after
                    for old, new in list(replacements.items()):
                        if new == before:
                            replacements[old] = after
                    changes += 1
            if not changes:
                break
        else:
            raise ValueError('Checksum graph did not converge')
        # CPU-only re-render; input summaries and metric values are unchanged.
        from research_runner.plots import render
        for name in ('configuration_stage2', 'benign'):
            directory = stage / name
            if (directory / 'summary.json').exists():
                dump(directory / 'figures.json', render(json.loads((directory / 'summary.json').read_text()), directory))
        # Establish fresh signatures in dependency order; no stale-results bypass.
        runner = ResearchRunner(config, store=ArtifactStore(stage, config))
        for name, parents in PARENTS:
            directory = stage / name
            if not directory.exists():
                continue
            manifest = json.loads((directory / 'manifest.json').read_text())
            before = dict(manifest)
            if name == 'configuration_stage2':
                completion = json.loads((directory / 'stage_complete.json').read_text())
                completion['stage1_manifest_sha256'] = sha(stage / 'configuration_stage1/manifest.json')
                dump(directory / 'stage_complete.json', completion)
            manifest.update(signature=runner.signature(name, parents), naming_version=2,
                files={p.relative_to(directory).as_posix(): sha(p) for p in directory.rglob('*')
                       if p.is_file() and p != directory / 'manifest.json'})
            dump(directory / 'manifest.json', manifest)
            record['signatures'][name] = {'before': before['signature'], 'after': manifest['signature'],
                                           'old_manifest_sha256': sha(saved / name / 'manifest.json')}
        # Compare every original scientific record, not just rounded summaries.
        count = 0
        for rel, entry in record['files'].items():
            before, after = saved / rel, stage / entry['new_path']
            if Path(rel).name == 'manifest.json' and len(Path(rel).parts) == 2:
                pass  # Root publication metadata was intentionally regenerated.
            elif Path(rel).name != 'figures.json':
                count += verify_pair(before, after)
            entry['after_sha256'] = sha(after)
        if source_hashes != {str(p.relative_to(root)): sha(p)
                             for p in (root / 'research_runner').rglob('*.py')}:
            raise ValueError('Runner source changed during migration; no results were published')
        for name, parents in PARENTS:
            if (stage / name).exists():
                runner.compatible(name, parents)
        record['verification'] = {'jsonl_rows_compared': count, 'numeric_and_boolean_equality': True,
                                  'opaque_inputs_and_ids_preserved': True, 'binary_bytes_preserved': True}
        dump(stage / '.naming_migration.json', record)
        if fail_before_publish:
            raise RuntimeError('Injected publication failure')
        recovery = target.parent / ('.naming-v1-recovery-' + str(time.time_ns()))
        os.replace(target, recovery)
        try:
            os.replace(stage, target)
        except BaseException:
            os.replace(recovery, target)
            raise
        record['recovery'] = str(recovery)
        dump(marker, record)
        dump(backup / 'migration_record.json', record)
        return {'status': 'migrated', 'record': str(marker), 'backup': str(backup),
                'recovery': str(recovery), 'verification': record['verification']}
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['artifacts'])
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Without this flag, report a dry-run only')
    args = parser.parse_args()
    print(json.dumps(migrate_artifacts(args.root.resolve(), args.backup.resolve(), dry_run=not args.apply), indent=2))


if __name__ == '__main__':
    main()
