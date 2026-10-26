"""Optional, verified Item 2 reuse for Item 6; missing matches fall back to inference."""
from copy import deepcopy
import csv
import hashlib
from pathlib import Path

from research_runner.artifacts import file_sha256, read_json, read_jsonl, write_json
from research_runner.progress import emit


class ThresholdReference:
    def __init__(self, runner, first):
        self.runner, self.first = runner, first
        self.targets = runner.config['experiments']['threshold'].get('reuse_configuration_recall_floors', [])
        self.full, self.unavailable = None, 'reuse_disabled'
        if self.targets:
            try:
                self.full = runner.compatible('configuration_stage2', ('configuration_stage1',))
                self.full_manifest = read_json(self.full / 'manifest.json')
                self.full_manifest_sha256 = file_sha256(self.full / 'manifest.json')
                self.first_manifest = read_json(first / 'manifest.json')
                self.first_manifest_sha256 = file_sha256(first / 'manifest.json')
                self.unavailable = None
            except (OSError, ValueError) as exc:
                self.full = None
                self.unavailable = f'item_2_unavailable_or_incompatible: {type(exc).__name__}: {exc}'
                emit('threshold.reuse', message=f'{self.unavailable}; matching points will run fresh inference.', force=True)

    def lookup(self, seed, order, rows, decisions, requests, calibration):
        if calibration.get('recall_floor') not in self.targets:
            return None, 'operating_point_not_selected_for_reuse'
        if self.full is None:
            return None, self.unavailable
        try:
            return self._load(seed, order, rows, decisions, requests, calibration), None
        except (OSError, ValueError, KeyError) as exc:
            return None, f'item_2_match_rejected: {type(exc).__name__}: {exc}'

    def _load(self, seed, order, rows, decisions, requests, calibration):
        from research_runner.backend_reuse import REPORT_FILES
        from research_runner.engine import resolve_pending
        full, first, runner = self.full, self.first, self.runner
        effective = read_json(full / 'effective_config.json')
        if effective['models']['audit'] != runner.config['models']['audit']:
            raise ValueError('Item 2 Audit model/prompt/schema/settings differ')
        if effective['runtime']['audit_python'] != runner.config['runtime']['audit_python']:
            raise ValueError('Item 2 Audit runtime differs')
        if read_json(full / 'stage_complete.json').get('stage1_manifest_sha256') != file_sha256(first / 'manifest.json'):
            raise ValueError('Item 2 Stage 2 belongs to a different Stage 1')
        checkpoint = read_json(first / 'scores/worker_result.json')['seeds'][str(seed)]['checkpoint']
        if calibration['threshold'] != checkpoint['gate_threshold']:
            raise ValueError('Gate threshold differs from Item 2')
        staged, original_requests, routing_path, request_path = runner._backend_inputs(first, seed, rows, order)
        if decisions != staged:
            raise ValueError('Current routing/scores differ from Item 2')
        stripped = [{k: v for k, v in item.items() if k != 'operating_point'} for item in requests]
        if stripped != original_requests:
            raise ValueError('Ordered request IDs/payloads differ from Item 2')
        directory = full / f'seed_{seed}/{order}'
        source = full / f'seed_{seed}/audit/aligned_responses.jsonl'
        answers = [item for item in read_jsonl(source) if item.get('configuration') == order]
        if [a.get('prefix_id') for a in answers] != [r['prefix_id'] for r in original_requests]:
            raise ValueError('Item 2 has missing, duplicate or misordered response IDs')
        expected = deepcopy(staged)
        lookup = {}
        for item, answer in zip(original_requests, answers):
            if (any(answer.get(k) != item[k] for k in ('request_id', 'payload', 'prefix_id', 'episode_id', 'seed', 'configuration'))
                    or type(answer.get('prediction')) is not int or answer['prediction'] not in (0, 1)
                    or answer.get('error') or answer.get('errors')
                    or hashlib.sha256(answer['payload'].encode()).hexdigest() != answer['request_id']):
                raise ValueError('Item 2 response payload/ID/seed/verdict mismatch')
            lookup[item['prefix_id']] = answer
        for decision in expected:
            if decision['pending_audit']:
                resolve_pending(decision, lookup[decision['id']])
        if expected != runner._saved_decisions(directory / 'prefix_decisions.jsonl', rows, order):
            raise ValueError('Item 2 final decisions do not match routing and responses')
        job_path = full / f'seed_{seed}/audit/worker/worker_job.json'
        if job_path.is_file():
            schema = read_json(job_path)['config']['models']['audit']['schema']
            if list(schema['properties']) != list(runner.config['models']['audit']['schema']['properties']):
                raise ValueError('Item 2 worker schema property order differs')
        sources = [directory / name for name in REPORT_FILES]
        sources += [source, routing_path, request_path, full / 'manifest.json', first / 'manifest.json',
                    full / 'effective_config.json', full / 'stage_complete.json', first / 'scores/worker_result.json']
        if job_path.is_file():
            sources.append(job_path)
        hashes = {}
        for path in sources:
            if path == full / 'manifest.json':
                expected_hash = self.full_manifest_sha256
            elif path == first / 'manifest.json':
                expected_hash = self.first_manifest_sha256
            elif path.is_relative_to(full):
                expected_hash = self.full_manifest['files'][path.relative_to(full).as_posix()]
            else:
                expected_hash = self.first_manifest['files'][path.relative_to(first).as_posix()]
            if file_sha256(path) != expected_hash:
                raise ValueError('Item 2 reference file changed after preflight')
            hashes[str(path)] = expected_hash
        return {'directory': directory, 'requests': requests,
                'answers': [{**answer, 'operating_point': item['operating_point']} for answer, item in zip(answers, requests)],
                'sources': hashes}


def publish_threshold_reference(bundle, condition):
    from research_runner.backend_reuse import publish_reference
    for name, digest in bundle['sources'].items():
        if file_sha256(name) != digest:
            raise ValueError('Item 2 reference changed after validation')
    publish_reference(bundle, condition)


def finish_threshold_summary(output):
    """Keep inherited measurements separate from fresh inference in reports."""
    output = Path(output)
    groups, records = {}, []
    for path in sorted(output.glob('seed_*/*/*/execution.json')):
        record = read_json(path)
        records.append(record)
        key = f"{record['operating_point']}/{record['configuration']}"
        groups.setdefault(key, []).append(record)
    summary = read_json(output / 'summary.json')
    for key, values in groups.items():
        reused = [r['seed'] for r in values if r['result_source'] == 'inherited_item_2']
        fresh = [r['seed'] for r in values if r['result_source'] == 'new_item_6_inference']
        summary[key]['execution_provenance'] = {'reused_seeds': reused, 'fresh_seeds': fresh,
                                               'new_audit_requests': sum(r['new_audit_requests'] for r in values),
                                               'latency_sources': sorted({r['latency_source'] for r in values})}
        summary[key]['new_inference_trials'] = sum(r['new_audit_requests'] > 0 for r in values)
        if reused:
            summary[key]['variation'] = ('inherited_item_2_seed_results_not_new_inference_trials' if not fresh
                                         else 'mixed_inherited_item_2_and_fresh_item_6; see execution_provenance')
    write_json(output / 'summary.json', summary)
    fields = ['seed', 'operating_point', 'configuration', 'result_source', 'latency_source',
              'new_audit_requests', 'reuse_fallback_reason']
    with (output / 'execution_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(records)
    write_json(output / 'execution_protocol.json', {'conditions': records,
               'timing_limitation': 'Inherited Item 2 timing uses its original mixed-configuration request schedule, not a new matched-workload measurement.'})
    figure_file = output / 'figures.json'
    if figure_file.is_file():
        figures = read_json(figure_file)
        figures['timing_provenance_note'] = 'Some points may inherit Item 2 timing; see execution_summary.csv before interpreting runtime comparisons.'
        write_json(figure_file, figures)
