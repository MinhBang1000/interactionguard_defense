"""Exact Item 2 reference reuse for the backend sweep, with no inference fallback."""
from copy import deepcopy
import hashlib
from pathlib import Path
import shutil

from research_runner.artifacts import file_sha256, read_json, read_jsonl, write_json, write_jsonl
from research_runner.engine import resolve_pending


REPORT_FILES = ('prefix_decisions.jsonl', 'metrics.json', 'timing.json', 'asr.json', 'episode_results.jsonl')


def validate_reference_candidate(config, candidate):
    allowed = {'name', 'reuse_configuration', 'vllm_version'}
    primary = config['models']['audit']
    if any(primary.get(key) != value for key, value in candidate.items() if key not in allowed):
        raise ValueError('Reused backend must exactly match the primary Item 2 Audit settings; no model/threshold/prompt override')
    if candidate.get('vllm_version') not in (None, '0.11.2'):
        raise ValueError('Reused Llama reference was measured with vLLM 0.11.2, not a new runtime')


def load_reference(runner, full, first, seed, rows, staged, requests):
    """Preflight every saved verdict against exact Stage 1 requests and routing."""
    order = runner.config['experiments']['backend']['order']
    effective = read_json(full / 'effective_config.json')
    if effective['models']['audit'] != runner.config['models']['audit']:
        raise ValueError('Item 2 Audit model/prompt/schema/settings do not match the reused backend')
    complete = read_json(full / 'stage_complete.json')
    if complete.get('stage1_manifest_sha256') != file_sha256(first / 'manifest.json'):
        raise ValueError('Reused Item 2 Stage 2 does not belong to the verified Stage 1')
    directory = full / f'seed_{seed}/{order}'
    for filename in REPORT_FILES:
        if not (directory / filename).is_file():
            raise ValueError(f'Missing Item 2 reference report: {directory / filename}; no inference fallback')
    decisions = runner._saved_decisions(directory / 'prefix_decisions.jsonl', rows, order)
    source = full / f'seed_{seed}/audit/aligned_responses.jsonl'
    if not source.is_file():
        raise ValueError(f'Missing Item 2 reference Audit responses: {source}; no inference fallback')
    answers = [item for item in read_jsonl(source) if item.get('configuration') == order]
    if [a.get('prefix_id') for a in answers] != [r['prefix_id'] for r in requests]:
        raise ValueError('Reused Item 2 responses have missing/duplicate/misordered prefix IDs')
    expected = deepcopy(staged)
    lookup = {}
    for item, answer in zip(requests, answers):
        if (any(answer.get(k) != item[k] for k in ('request_id', 'payload', 'prefix_id', 'episode_id', 'seed', 'configuration'))
                or type(answer.get('prediction')) is not int or answer['prediction'] not in (0, 1)
                or answer.get('error') or answer.get('errors')
                or hashlib.sha256(answer['payload'].encode()).hexdigest() != answer['request_id']):
            raise ValueError('Reused Item 2 Audit response payload/ID/verdict mismatch')
        lookup[item['prefix_id']] = answer
    for decision in expected:
        if decision['pending_audit']:
            resolve_pending(decision, lookup[decision['id']])
    if expected != decisions:
        raise ValueError('Item 2 reference decisions do not match Stage 1 routing + the saved Audit responses')
    return {'directory': directory, 'decisions': decisions, 'requests': requests, 'answers': answers,
            'sources': {**{str(directory / filename): file_sha256(directory / filename) for filename in REPORT_FILES},
                        str(source): file_sha256(source), str(full / 'manifest.json'): file_sha256(full / 'manifest.json')}}


def publish_reference(bundle, directory):
    """Copy original scientific reports byte-for-byte; write provenance separately."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for filename in REPORT_FILES:
        source = bundle['directory'] / filename
        expected = bundle['sources'][str(source)]
        if file_sha256(source) != expected:
            raise ValueError('Item 2 reference changed after preflight; refusing reuse')
        shutil.copyfile(source, directory / filename)
        if file_sha256(directory / filename) != expected:
            raise ValueError('Item 2 reference copy checksum mismatch')
    write_jsonl(directory / 'audit/requests.jsonl', bundle['requests'])
    write_jsonl(directory / 'audit/aligned_responses.jsonl', bundle['answers'])
    provenance = {'source_experiment': 'configuration_stage2', 'source': str(bundle['directory']),
                  'source_sha256': bundle['sources'], 'verdicts_reused': True, 'new_inference_requests': 0,
                  'new_latency_measurement': False, 'latency_inherited_from_item_2': True,
                  'not_an_independent_inference_trial': True,
                  'timing_limitation': 'Item 2 used its original mixed-configuration request schedule; inherited timing is not a fresh matched-workload backend benchmark.'}
    write_json(directory / 'reuse.json', provenance)
    write_json(directory / 'audit/timing.json', {**provenance, 'uncached_measured': False,
                                               'submitted_requests': 0, 'reused_requests': len(bundle['answers'])})
