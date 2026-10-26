"""Opt-in, manually selected case stress test; no dataset mutation or ASR claims."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import re

from research_runner.artifacts import (
    ArtifactStore, file_sha256, fingerprint, read_json, read_jsonl, write_json, write_jsonl,
)
from research_runner.payloads import SECTION, parse, build_payload, request
from research_runner.progress import emit
from research_runner.scoring import route


ORDER = 'Screen-Gate-Audit'
PARENTS = ('dataset', 'training', 'configuration_stage1', 'configuration_stage2')


def validate_settings(config):
    settings = config['experiments']['auditor_targeted']
    allowed = {'seed', 'case_id', 'order', 'channel', 'block_index', 'json_field',
               'payload_anchor', 'templates', 'resume'}
    if set(settings) - allowed:
        raise ValueError('Unknown case-test settings')
    if type(settings.get('seed')) is not int or settings['seed'] < 0:
        raise ValueError('seed must be a nonnegative integer')
    if settings.get('order') != ORDER:
        raise ValueError(f'order must be {ORDER}')
    if settings.get('channel') not in {'prompt', 'retrieval', 'tool'}:
        raise ValueError('channel must be prompt, retrieval or tool')
    if type(settings.get('block_index')) is not int or settings['block_index'] < 0:
        raise ValueError('block_index must be a nonnegative integer')
    if settings.get('json_field') is not None and (
            not isinstance(settings['json_field'], str) or not settings['json_field'].strip()):
        raise ValueError('json_field must be null or a nonempty top-level JSON string field')
    for key in ('case_id', 'payload_anchor'):
        if not isinstance(settings.get(key), str) or not settings[key].strip():
            raise ValueError(f'{key} must be nonempty')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', settings['case_id']):
        raise ValueError('case_id must be a safe prefix ID, not a path')
    templates = settings.get('templates')
    if not isinstance(templates, list) or len(templates) != 2:
        raise ValueError('This small case test requires exactly two predeclared templates')
    names = set()
    for template in templates:
        if not isinstance(template, dict) or set(template) != {'name', 'text'}:
            raise ValueError('Each template requires only name and text')
        name = template['name']
        if (not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', name)
                or name == 'clean' or name in names):
            raise ValueError('Template names must be unique safe identifiers other than clean')
        names.add(name)
        if not isinstance(template['text'], str) or not template['text'].strip():
            raise ValueError('Template text must be nonempty')
        if SECTION.search(template['text']):
            raise ValueError('Template must stay inside data content, not create real channel boundaries')
    if settings.get('resume', False):
        raise ValueError('Case tests rerun all three versions; use overwrite, not cached resume')


def _channel(tag):
    return 'tool' if tag.startswith('TOOL:') or tag == 'TOOLS' else {
        'PROMPT': 'prompt', 'RETRIEVE': 'retrieval'}.get(tag)


def _visible(text, payload):
    return text in payload or json.dumps(text, ensure_ascii=False)[1:-1] in payload


def make_variants(original, settings, payload_settings):
    """Edit exactly one explicitly selected data block and retain JSON tool shape."""
    sequence = original['sequence']
    matches = list(SECTION.finditer(sequence))
    selected = [(i, match) for i, match in enumerate(matches)
                if _channel(match.group(1)) == settings['channel']]
    index = settings['block_index']
    if index >= len(selected):
        raise ValueError('Selected injection channel/block does not exist in this prefix')
    position, match = selected[index]
    end = matches[position + 1].start() if position + 1 < len(matches) else len(sequence)
    block = sequence[match.end():end].strip()
    field = settings.get('json_field')
    value = json.loads(block) if field is not None else None
    if field is not None and (not isinstance(value, dict) or not isinstance(value.get(field), str)):
        raise ValueError('Selected JSON field is missing or is not a string')
    content = value[field] if field is not None else block
    anchor = settings['payload_anchor']
    if content.count(anchor) != 1:
        raise ValueError('payload_anchor must occur exactly once in the selected payload content')
    clean_payload = build_payload(sequence, payload_settings)
    if not _visible(anchor, clean_payload):
        raise ValueError('Original payload anchor is not visible to Audit after clipping')
    variants = []
    for name, text in [('clean', None)] + [(t['name'], t['text']) for t in settings['templates']]:
        row = deepcopy(original)
        if text is not None:
            modified = content.replace(anchor, anchor + '\n\n' + text, 1)
            if field is None:
                updated = modified
            else:
                obj = deepcopy(value)
                obj[field] = modified
                updated = json.dumps(obj, ensure_ascii=False)
            # Preserve all unrelated blocks byte-for-byte, including separators.
            start = match.end() + len(sequence[match.end():end]) - len(sequence[match.end():end].lstrip())
            row['sequence'] = sequence[:start] + updated + sequence[start + len(block):]
            actual = build_payload(row['sequence'], payload_settings)
            if not _visible(anchor + '\n\n' + text, actual):
                raise ValueError(f'{name}: payload or attack template was clipped; choose another explicit case/anchor')
            if actual == clean_payload:
                raise ValueError('Mutation did not reach the final Audit input')
            if field is not None:
                parsed = parse(row['sequence'])['TOOLS' if settings['channel'] == 'tool' else
                                               'PROMPT' if settings['channel'] == 'prompt' else 'RETRIEVE']
                if json.loads(parsed[index])[field] != modified:
                    raise ValueError('JSON payload did not survive serialization')
            # Downstream scorers consume sequence; keep the copied channel view consistent too.
            channels = deepcopy(row.get('channels', {}))
            if channels:
                if settings['channel'] == 'prompt':
                    channels['prompt'] = updated
                elif settings['channel'] == 'tool':
                    channels['tools'][index]['content'] = updated
                else:
                    channels['retrieval'][index] = updated
                row['channels'] = channels
        row.update(id=f"{original['id']}__{name}", source_prefix_id=original['id'],
                   stress_variant=name, injected_text=text, stress_test_only=True)
        variants.append(row)
    return variants


def _one(path, key, value):
    found = [row for row in read_jsonl(path) if row.get(key) == value]
    if len(found) != 1:
        raise ValueError(f'Expected exactly one {key}={value} in {path}; no automatic case search')
    return found[0]


def prepare_case(runner):
    """Read-only prerequisite/case checks, also usable without any model call."""
    validate_settings(runner.config)
    settings = runner.config['experiments']['auditor_targeted']
    if settings['seed'] not in runner.config['training']['seeds']:
        raise ValueError('Case-test seed is not a configured training seed; select an existing Item 2 seed')
    data = runner.data_root()
    training = runner.compatible('training', ('dataset',))
    first = runner.compatible('configuration_stage1', ('dataset', 'training'))
    second = runner.compatible('configuration_stage2', ('configuration_stage1',))
    seed, id_ = settings['seed'], settings['case_id']
    row = _one(data / 'processed/test_processed.jsonl', 'id', id_)
    reference_path = second / f'seed_{seed}' / ORDER / 'prefix_decisions.jsonl'
    reference = _one(reference_path, 'id', id_)
    if (row['label'] != 1 or reference.get('label') != 1 or reference.get('prediction') != 1
            or reference.get('pending_audit') or not reference.get('route')
            or reference['route'][-1] != {'layer': 'Audit', 'action': 'Stop'}
            or reference.get('audit', {}).get('error') or reference.get('audit', {}).get('prediction') != 1):
        raise ValueError('Selected case must be a malicious prefix previously stopped by Audit in Item 2')
    if reference.get('original_trace_id') != row['original_trace_id']:
        raise ValueError('Reference episode ID does not match the selected prefix')
    if reference['audit'].get('payload') != build_payload(row['sequence'], runner.config['models']['audit']['payload']):
        raise ValueError('Reference Audit payload differs from the current input builder')
    checkpoint = read_json(first / 'scores/worker_result.json')['seeds'][str(seed)]['checkpoint']
    if checkpoint != read_json(training / 'checkpoints.json')[str(seed)]:
        raise ValueError('Stage 1 and training checkpoints differ')
    episode = _one(data / 'splits/test.jsonl', 'id', row['original_trace_id'])
    variants = make_variants(row, settings, runner.config['models']['audit']['payload'])
    provenance = {'case_id': id_, 'episode_id': row['original_trace_id'], 'seed': seed, 'order': ORDER,
                  'checkpoint': checkpoint,
                  'gate_threshold': checkpoint['gate_threshold'],
                  'screen_threshold': read_json(first / 'scores/screen_calibration.json')['threshold'],
                  'audit_threshold': runner.config['models']['audit']['threshold'],
                  'audit_config_sha256': fingerprint(runner.config['models']['audit']),
                  'settings_sha256': fingerprint(settings),
                  'source_files': {str(path): file_sha256(path) for path in (
                      data / 'processed/test_processed.jsonl', data / 'splits/test.jsonl', reference_path)},
                  'parent_manifests': {name: file_sha256(path / 'manifest.json') for name, path in
                      [('dataset', data), ('training', training), ('configuration_stage1', first),
                       ('configuration_stage2', second)]},
                  'original_sequence_sha256': fingerprint(row['sequence']),
                  'last_safe_prefix_step': row.get('last_safe_prefix_step'),
                  'step_number': row['step_number'],
                  'scope': 'single_manually_selected_prefix_case_not_population_metrics_or_episode_ASR'}
    return row, episode, reference, variants, provenance


def run_case(runner, *, overwrite=False):
    from research_runner.engine import index_scores, resolve_pending

    original, episode, reference, variants, provenance = prepare_case(runner)
    settings = runner.config['experiments']['auditor_targeted']
    final = runner.store.root / 'auditor_targeted'
    # Copies live outside raw/splits/processed/training and never enter dataset exports.
    copy_root = Path(runner.config['data']['root']) / 'auditor_targeted'
    if copy_root.is_symlink():
        raise ValueError('Case data directory must not be a symlink')
    data_store = ArtifactStore(copy_root, runner.config)
    copy_name = f"seed_{settings['seed']}/{settings['case_id']}"
    signature = runner.signature('auditor_targeted', PARENTS)
    input_signature = fingerprint({'provenance': provenance, 'variants': variants})
    runner.store.ensure('auditor_targeted', signature, overwrite=overwrite)
    data_store.ensure(copy_name, input_signature, overwrite=overwrite)
    emit('auditor_targeted.copy', 0, 3, unit='versions', message='Copying the explicit case only; original dataset remains read-only.', force=True)
    with data_store.transaction(copy_name, input_signature, overwrite=overwrite) as stage:
        write_jsonl(stage / 'original_prefix.jsonl', [original])
        write_jsonl(stage / 'original_episode.jsonl', [episode])
        write_json(stage / 'reference_decision.json', reference)
        write_jsonl(stage / 'variants.jsonl', variants)
        write_json(stage / 'provenance.json', provenance)
        write_json(stage / 'templates.json', settings['templates'])
        write_jsonl(stage / 'audit_input_previews.jsonl', [request(row, runner.config['models']['audit']['payload']) for row in variants])
    copied = data_store.load(copy_name, input_signature)
    emit('auditor_targeted.copy', 3, 3, unit='versions', status='completed', message=str(copied), force=True)
    with runner.transaction('auditor_targeted', PARENTS, overwrite) as out:
        # A fresh score worker exits before the Audit worker is started.
        emit('auditor_targeted.score', message='Fresh Screen/Gate scores on three versions; no calibration or training.', force=True)
        runner.executor.run('auditor_targeted_score', {'rows': str(copied / 'variants.jsonl'),
                            'checkpoint': provenance['checkpoint']}, out / 'scores')
        gate = index_scores(read_jsonl(out / 'scores/gate_scores.jsonl'), variants)
        screen = index_scores(read_jsonl(out / 'scores/screen_scores.jsonl'), variants)
        decisions = route(variants, gate, screen, ORDER.split('-'), provenance['gate_threshold'], provenance['screen_threshold'])
        pending = {d['id'] for d in decisions if d['pending_audit']}
        requests = [{**request(row, runner.config['models']['audit']['payload']), 'seed': settings['seed'],
                     'configuration': ORDER, 'variant': row['stress_variant']} for row in variants if row['id'] in pending]
        emit('auditor_targeted.audit', 0, len(requests), unit='requests', message='Score worker completed; Audit runs only for escalated variants.', force=True)
        answers = {a['prefix_id']: a for a in runner.infer(requests, out / 'audit')}
        for decision in decisions:
            if decision['pending_audit']:
                resolve_pending(decision, answers[decision['id']])
        write_jsonl(out / 'prefix_decisions.jsonl', decisions)
        table = []
        for row, decision in zip(variants, decisions):
            actions = {step['layer']: step['action'] for step in decision['route']}
            table.append({'variant': row['stress_variant'], 'prefix_id': row['id'],
                          'screen_score': screen[row['id']]['score'], 'screen_action': actions['Screen'],
                          'gate_score': gate[row['id']]['score'], 'gate_action': actions.get('Gate', 'Not called'),
                          'audit_action': actions.get('Audit', 'Not called'),
                          'audit_confidence': decision.get('audit', {}).get('score'),
                          'final_decision': 'Stop' if decision['prediction'] else 'Allow',
                          'auditor_called': 'Audit' in actions})
            emit('auditor_targeted.verdict', message=f"{row['stress_variant']}: Screen={actions['Screen']}; "
                 f"Gate={actions.get('Gate', 'Not called')}; Audit={actions.get('Audit', 'Not called')}; "
                 f"Final={table[-1]['final_decision']}", force=True)
        clean_reproduced = table[0]['final_decision'] == 'Stop' and table[0]['audit_action'] == 'Stop'
        summary = {'case_id': settings['case_id'], 'seed': settings['seed'], 'order': ORDER,
                   'clean_control_reproduced_audit_stop': clean_reproduced, 'versions': table,
                   'case_data_directory': str(copied), 'case_data_manifest_sha256': file_sha256(copied / 'manifest.json'),
                   'scope': provenance['scope'], 'no_ASR_or_F1_computed': True,
                   'caution': 'Not called does not establish auditor robustness; if clean control differs, do not attribute changes solely to injection.'}
        write_json(out / 'summary.json', summary)
        write_json(out / 'provenance.json', provenance)
        with (out / 'comparison.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(table[0]))
            writer.writeheader()
            writer.writerows(table)
        lines = ['# Auditor-targeted case stress test', '',
                 f"Case: {settings['case_id']}; Gate seed: {settings['seed']}; clean Audit Stop reproduced: {clean_reproduced}.", '',
                 '| Version | Screen | Gate | Audit | Final |', '| --- | --- | --- | --- | --- |']
        lines.extend(f"| {r['variant']} | {r['screen_action']} | {r['gate_action']} | {r['audit_action']} | {r['final_decision']} |" for r in table)
        lines += ['', 'Case-based observations only: no F1, population bypass rate, or episode ASR.',
                  'An upstream Stop is cascade prevention, not evidence that Audit resisted the injected instruction.']
        (out / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        write_json(out / 'stage_complete.json', {'versions': 3, 'inference_requests': len(requests), 'case_study_only': True})
        emit('auditor_targeted.complete', 3, 3, unit='versions', status='completed', message=f'Comparison: {final / "REPORT.md"}', force=True)
        from research_runner.engine import rebase_json
        rebase_json(out, final)
    return runner.prerequisite('auditor_targeted')
