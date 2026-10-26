"""Experiment orchestration. Models live only in blocking subprocess workers."""
from collections import defaultdict
from copy import deepcopy
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

from research_runner.artifacts import IncompatibleResultsError, read_json, read_jsonl, write_json, write_jsonl, file_sha256, fingerprint
from research_runner.execution import ProcessExecutor
from research_runner.payloads import request
from research_runner.scoring import route, calibrate_gate, metrics, score_episodes
from research_runner.progress import Reporter, emit, iterable, scope
from research_runner.threshold_sweep import select_gate_threshold

NAMES = {0: 'dataset', 1: 'training', 2: 'configuration', 3: 'baselines',
         4: 'channel_ablation', 5: 'backend', 6: 'threshold', 7: 'benign',
         8: 'auditor_targeted'}


def index_scores(records, rows):
    indexed = {r['id']: r for r in records}
    if len(indexed) != len(records) or set(indexed) != {r['id'] for r in rows}:
        raise ValueError('Score IDs missing, duplicated or from a different dataset')
    return indexed


def rebase_json(root, destination):
    """Published paths must not point into a now-renamed staging directory."""
    old, new = str(Path(root).resolve()), str(Path(destination).resolve())
    def replace(value):
        if isinstance(value, str):
            return new + value[len(old):] if value == old or value.startswith(old + '/') else value
        if isinstance(value, list):
            return [replace(v) for v in value]
        if isinstance(value, dict):
            return {k: replace(v) for k, v in value.items()}
        return value
    for path in Path(root).rglob('*.json'):
        # Model directories are content-addressed. Rewriting training metadata
        # would invalidate their saved encoder fingerprint after publication.
        if 'encoder' in path.relative_to(root).parts:
            continue
        write_json(path, replace(read_json(path)))


def resolve_pending(decision, answer):
    if not decision['pending_audit']:
        raise ValueError('Cannot attach Audit to a prefix not escalated to Audit')
    decision.update(prediction=answer['prediction'], pending_audit=False, audit=answer)
    decision['layer_scores']['Audit'] = answer
    if decision['route'][-1] != {'layer': 'Audit', 'action': 'Pending'}:
        raise ValueError('Pending Audit routing record is inconsistent')
    decision['route'][-1] = {'layer': 'Audit', 'action': 'Stop' if answer['prediction'] else 'Forward'}


class ResearchRunner:
    def __init__(self, config, store=None, executor=None):
        from research_runner.artifacts import ArtifactStore
        self.config = config
        self.store = store or ArtifactStore(Path(config['paths']['results']), config)
        self.executor = executor or ProcessExecutor(config)

    def signature(self, name, parents=()):
        # Changing a backend candidate must not invalidate expensive training.
        exp_name = 'configuration' if name.startswith('configuration_') else name
        experiment = deepcopy(self.config['experiments'].get(exp_name, {}))
        for key in ('resume', 'stage'):
            experiment.pop(key, None)
        models = self.config['models']
        scientific = {'version': self.config['version'], 'naming': self.config['naming'], 'experiment': experiment}
        if name in {'baselines', 'backend', 'threshold'}:
            scientific['primary_order'] = self.config['primary_order']
        if name == 'dataset':
            scientific['data'] = self.config['data']
            raw = Path(self.config['data']['root']) / 'raw'
            scientific['raw'] = {p.name: file_sha256(p) for p in sorted(raw.glob('*.jsonl'))}
            scientific['outcome_inputs'] = {key: file_sha256(self.config['data'][key])
                if self.config['data'].get(key) and Path(self.config['data'][key]).is_file() else None
                for key in ('judges', 'reviews')}
            from research_runner.implementation.data.data_split import load_membership
            scientific['membership'] = load_membership(Path(self.config['data']['root']), None)
        elif name == 'training':
            scientific.update(training=self.config['training'], encoder=models['encoder'], autoencoder=models['autoencoder'])
        else:
            scientific['seeds'] = self.config['training']['seeds']
            if name == 'configuration_stage1':
                scientific.update(screen=models['screen'], calibration=self.config['training']['calibration'], payload=models['audit']['payload'])
            elif name != 'baselines':
                scientific.update(audit=models['audit'], timing=self.config['experiments']['timing'])
        runtime = self.config['runtime']
        if name in ('training', 'configuration_stage1'):
            scientific['runtime'] = {'python': runtime['python'], 'device': runtime['device']}
        elif name != 'dataset':
            scientific['runtime'] = {'audit_python': runtime['audit_python'], 'device': runtime['device']}
        sources = {}
        workspace = Path(self.config['paths']['workspace'])
        for directory in ('research_runner',):
            for parent, directories, files in os.walk(workspace / directory):
                directories[:] = sorted(d for d in directories if d not in {'.venv', 'venv', 'env', '__pycache__', '.git'})
                for name_ in sorted(files):
                    path = Path(parent) / name_
                    if path.suffix == '.py':
                        sources[str(path.relative_to(workspace))] = file_sha256(path)
        return fingerprint({'name': name, 'config': scientific, 'sources': sources,
                            'parents': {p: file_sha256(self.prerequisite(p) / 'manifest.json') for p in parents}})

    def prerequisite(self, name):
        try:
            return self.store.load(name)
        except (FileNotFoundError, ValueError) as exc:
            number = {'dataset': 0, 'training': 1, 'configuration_stage1': 2, 'configuration_stage2': 2}.get(name)
            raise ValueError(f'Missing or incompatible prerequisite: {name}. Run Experiment {number} first. {exc}') from exc

    def compatible(self, name, parents=()):
        required = {'training': ('dataset',), 'configuration_stage1': ('dataset', 'training'),
                    'configuration_stage2': ('configuration_stage1',)}
        for parent in parents:
            self.compatible(parent, required.get(parent, ()))
        expected = self.signature(name, parents)
        try:
            return self.store.load(name, expected)
        except IncompatibleResultsError:
            # Explicit, audited code-only upgrades may preserve immutable inputs.
            # Never infer compatibility just from matching configuration names.
            receipt_path = self.store.root / '.compatibility' / f'{name}.json'
            if receipt_path.is_symlink() or not receipt_path.is_file():
                raise
            receipt = read_json(receipt_path)
            actual = self.store.load(name)
            manifest = read_json(actual / 'manifest.json')
            if (not isinstance(receipt, dict) or receipt.get('version') != 1
                    or receipt.get('name') != name or receipt.get('parents') != list(parents)
                    or receipt.get('from_signature') != manifest['signature']
                    or receipt.get('to_signature') != expected
                    or receipt.get('manifest_sha256') != file_sha256(actual / 'manifest.json')):
                raise IncompatibleResultsError(f'Upgrade receipt does not match current inputs/code: {name}')
            emit('artifact.compatibility', message=f'Using audited code-upgrade receipt for {name}; original results unchanged.')
            return actual

    def data_root(self):
        path = self.compatible('dataset')
        report = read_json(path / 'dataset_report.json')
        for filename, expected in report.get('raw_sha256', {}).items():
            if file_sha256(Path(self.config['data']['root']) / 'raw' / filename) != expected:
                raise ValueError('Raw dataset changed. Run Experiment 0 again before downstream experiments.')
        return path

    def rows(self, split='test'):
        return read_jsonl(self.data_root() / 'processed' / f'{split}_processed.jsonl')

    def transaction(self, name, parents, overwrite, resume=False):
        return self.store.transaction(name, self.signature(name, parents), overwrite=overwrite, resume=resume and not overwrite)

    def run(self, item, *, overwrite=False):
        if item not in {*NAMES, 9}:
            raise ValueError('Unknown experiment selection')
        name = 'backup' if item == 9 else NAMES[item]
        with scope(Path(self.config['paths']['results']) / '.progress' / name, f'Experiment {item}: {name}'):
            if item == 9:
                from research_runner.artifacts import backup_snapshot
                return backup_snapshot(self.config)
            return getattr(self, name)(overwrite=overwrite)

    def cancel(self, item):
        name = 'backup' if item == 9 else NAMES[item]
        Reporter(Path(self.config['paths']['results']) / '.progress' / name).update(
            f'Experiment {item}: {name}', status='cancelled',
            message='Overwrite declined; existing scientific results are unchanged.', force=True)

    def dataset(self, *, overwrite=False):
        from research_runner.dataset import prepare_dataset
        with self.transaction('dataset', (), overwrite) as out:
            config = deepcopy(self.config)
            config['_dataset_final'] = str(Path(self.config['paths']['results']) / 'dataset')
            report = prepare_dataset(config, out)
            write_json(out / 'dataset_report.json', report)
            # Preparation already writes final provenance paths before hashing.
            # Re-serializing metadata here would invalidate its nested checksums.
        return self.prerequisite('dataset')

    def auditor_targeted(self, *, overwrite=False):
        from research_runner.auditor_targeted import run_case
        return run_case(self, overwrite=overwrite)

    def training(self, *, overwrite=False):
        data = self.data_root()
        config = deepcopy(self.config)
        config['_prepared_root'] = str(data / 'training')
        # Workers receive an immutable runtime snapshot, not legacy defaults.
        executor = self.executor
        if isinstance(executor, ProcessExecutor):
            executor = ProcessExecutor(config)
        with self.transaction('training', ('dataset',), overwrite) as out:
            checkpoints = {}
            for seed in iterable(self.config['training']['seeds'], 'training.seeds', unit='seeds'):
                emit(f'training.seed_{seed}', message='Encoder training, AE selection and validation calibration.', force=True)
                checkpoints[str(seed)] = executor.run('train', {'seed': seed}, out / f'seed_{seed}')
            write_json(out / 'checkpoints.json', checkpoints)
            rebase_json(out, Path(self.config['paths']['results']) / 'training')
        return self.prerequisite('training')

    def configuration(self, *, overwrite=False):
        stage = self.config['experiments']['configuration']['stage']
        result = None
        if stage in ('stage1', 'full'):
            emit('configuration.stage1', message='Gate/Screen scores and routing. Audit is not loaded.', force=True)
            result = self.configuration_stage1(overwrite=overwrite)
        if stage in ('stage2', 'full'):
            emit('configuration.stage2', message='Stage 1 process has exited; validating artifacts before starting Audit.', force=True)
            result = self.configuration_stage2(overwrite=overwrite)
        return result

    def configuration_stage1(self, *, overwrite=False):
        data, training = self.data_root(), self.compatible('training', ('dataset',))
        settings = self.config['experiments']['configuration']
        with self.transaction('configuration_stage1', ('dataset', 'training'), overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            result = self.executor.run('score', {'rows': str(data / 'processed/test_processed.jsonl'),
                                      'validation': str(data / 'training/encoder/val.jsonl'),
                                      'checkpoints': read_json(training / 'checkpoints.json')}, out / 'scores')
            rows = self.rows()
            screen = index_scores(read_jsonl(out / 'scores/screen_scores.jsonl'), rows)
            for seed in iterable(self.config['training']['seeds'], 'stage1.seeds', unit='seeds'):
                checkpoint = result['seeds'][str(seed)]['checkpoint']
                gate = index_scores(read_jsonl(out / f'scores/gate_scores_seed{seed}.jsonl'), rows)
                requests = []
                for order in iterable(settings['orders'], f'stage1.seed_{seed}.routing', unit='configurations'):
                    emit(f'stage1.seed_{seed}.{order}', message='Computing routing and saving pending Audit requests.', force=True)
                    decisions = route(rows, gate, screen, order.split('-'), checkpoint['gate_threshold'], result['screen_calibration']['threshold'])
                    write_jsonl(out / f'seed_{seed}/{order}/routing.jsonl', decisions)
                    pending = {d['id'] for d in decisions if d['pending_audit']}
                    for row in rows:
                        if row['id'] in pending:
                            req = request(row, self.config['models']['audit']['payload'])
                            requests.append({**req, 'configuration': order, 'seed': seed})
                write_jsonl(out / f'seed_{seed}/audit_requests.jsonl', requests)
            write_json(out / 'stage_complete.json', {'stage': 1, 'score_worker_completed': True,
                                                     'screen_threshold': result['screen_calibration']['threshold']})
            rebase_json(out, Path(self.config['paths']['results']) / 'configuration_stage1')
        return self.prerequisite('configuration_stage1')

    def infer(self, requests, out, backend=None):
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        write_jsonl(out / 'requests.jsonl', requests)
        unique = list({r['request_id']: r for r in requests}.values())
        if any(hashlib.sha256(r['payload'].encode()).hexdigest() != r['request_id'] for r in requests):
            raise ValueError('Request payload hash mismatch')
        write_jsonl(out / 'unique_requests.jsonl', unique)
        # Accuracy deduplication is kept separate from uncached timing trials.
        measure = self.config['experiments']['timing']['measure_uncached']
        deduplicate = self.config['experiments']['timing']['deduplicate_accuracy_requests'] and not measure
        selected = unique if deduplicate else requests
        submitted = [{**r, 'request_id': f'{i}:{r["request_id"]}'} for i, r in enumerate(selected)]
        emit('audit.submission', 0, len(submitted), unit='requests',
             message=f'{len(requests)} routed requests; {len(unique)} unique payloads. Output: {out}', force=True)
        write_jsonl(out / 'submitted_requests.jsonl', submitted)
        if submitted:
            self.executor.run('infer', {'requests': str(out / 'submitted_requests.jsonl'), 'backend': backend}, out / 'worker', llm=True)
            responses = read_jsonl(out / 'worker/responses.jsonl')
        else:
            responses = []
        expected = {r['request_id'] for r in submitted}
        got = {r['request_id'] for r in responses}
        if got != expected or len(got) != len(responses):
            raise ValueError('Audit responses missing, duplicated or from another request set')
        if any(type(r.get('prediction')) is not int or r['prediction'] not in (0, 1) or r.get('error') for r in responses):
            write_jsonl(out / 'failed_responses.jsonl', responses)
            raise ValueError('Audit inference failed or returned invalid verdicts; errors are not benign predictions.')
        by_id = {r['request_id']: r for r in responses}
        aligned = []
        if not deduplicate:
            aligned = [{**r, **by_id[s['request_id']], 'request_id': r['request_id']} for r, s in zip(requests, submitted)]
        else:
            lookup = {r['request_id']: by_id[s['request_id']] for r, s in zip(unique, submitted)}
            aligned = [{**r, **lookup[r['request_id']], 'request_id': r['request_id']} for r in requests]
        write_jsonl(out / 'aligned_responses.jsonl', aligned)
        emit('audit.submission', len(submitted), len(submitted), unit='requests',
             message='All response IDs validated and aligned.', status='completed', force=True)
        write_json(out / 'timing.json', {'uncached_measured': measure, 'submitted_requests': len(submitted),
                                        'accuracy_unique_payloads': len(unique), 'cache_used': False,
                                        'interpretation': 'observed_AUDIT_request_or_batch_time; staged_pipeline_is_component_estimate'})
        return aligned

    def report(self, out, rows, decisions, *, episode_level=True):
        out = Path(out)
        emit('report', message=f'Metrics, timing and ASR: {out}', force=True)
        if [r['id'] for r in rows] != [d['id'] for d in decisions]:
            raise ValueError('Report decisions are not aligned with sample IDs')
        write_jsonl(out / 'prefix_decisions.jsonl', decisions)
        report = metrics(decisions)
        report['fnr'] = report['fn'] / (report['tp'] + report['fn']) if report['tp'] + report['fn'] else 0.
        report['per_category'] = {}
        for category in sorted({r.get('attack_type') or 'benign' for r in rows}):
            chosen = [(r, d) for r, d in zip(rows, decisions) if (r.get('attack_type') or 'benign') == category]
            report['per_category'][category] = metrics([d for _, d in chosen])
        report['samples'] = len(rows)
        report['sample_ids'] = [r['id'] for r in rows]
        traffic = {layer: sum(any(step.get('layer') == layer for step in d.get('route', [])) for d in decisions)
                   for layer in ('Gate', 'Screen', 'Audit')}
        report['processed_counts'] = traffic
        report['processed_ratios'] = {key: value / len(rows) if rows else 0. for key, value in traffic.items()}
        components = {}
        for layer in ('Gate', 'Screen', 'Audit'):
            values = []
            for d in decisions:
                if layer == 'Audit' and d.get('channel_results'):
                    values.append(sum(float(a.get('seconds', 0.)) for a in d['channel_results']))
                    continue
                record = d.get('audit', {}) if layer == 'Audit' else d.get('layer_scores', {}).get(layer, {})
                if layer == 'Audit' and not record:
                    record = d.get('layer_scores', {}).get('Audit', {})
                values.append(float(record.get('seconds', 0.)))
            components[layer] = sum(values)
        if any(d.get('detector_details') or d.get('channel_verdicts') for d in decisions):
            components['baseline'] = sum(float(d.get('detector_details', {}).get('seconds', 0.)) +
                                        sum(float(value['detector_details'].get('seconds', 0.)) for value in d.get('channel_verdicts', {}).values())
                                        for d in decisions)
        report['estimated_pipeline_seconds_per_sample'] = sum(components.values()) / len(rows) if rows else 0.
        write_json(out / 'timing.json', {'component_seconds': components,
                   'estimated_pipeline_seconds_per_sample': sum(components.values()) / len(rows) if rows else 0.,
                   'interpretation': 'component_based_pipeline_estimate_not_direct_end_to_end',
                   'model_loading_excluded': True,
                   'not_a_direct_online_or_production_latency_measurement': True})
        write_json(out / 'metrics.json', report)
        if episode_level:
            data = self.data_root()
            episodes = read_jsonl(data / 'splits/test.jsonl')
            from research_runner.implementation.evaluation.asr import load_judges
            asr = score_episodes(episodes, decisions, load_judges(data / 'judges/prompt_success_judgements.jsonl'))
            write_json(out / 'asr.json', asr)
            write_jsonl(out / 'episode_results.jsonl', asr['episodes'])
        return report

    def _saved_decisions(self, source, rows, order, *, allow_pending=False):
        if not source.is_file():
            raise ValueError(f'Missing Experiment 2 configuration {order}: {source}. Run the required configuration explicitly.')
        decisions = read_jsonl(source)
        if [d['id'] for d in decisions] != [r['id'] for r in rows]:
            raise ValueError(f'Experiment 2 IDs are missing, duplicated or misordered: {source}')
        layers = order.split('-')
        for row, decision in zip(rows, decisions):
            if any(decision.get(k) != row[k] for k in ('label', 'original_trace_id', 'step_number')):
                raise ValueError(f'Experiment 2 sample metadata mismatch: {source}/{row["id"]}')
            pending = decision.get('pending_audit')
            steps = decision.get('route', [])
            if (type(pending) is not bool or not steps
                    or [s.get('layer') for s in steps] != layers[:len(steps)]
                    or any(s.get('action') != 'Check' for s in steps[:-1])):
                raise ValueError(f'Invalid saved cascade routing: {source}/{row["id"]}')
            if pending:
                if not allow_pending or steps[-1] != {'layer': 'Audit', 'action': 'Pending'} or decision.get('prediction') is not None:
                    raise ValueError(f'Unresolved or inconsistent Experiment 2 verdict: {source}')
            elif (type(decision.get('prediction')) is not int or decision['prediction'] not in (0, 1)
                  or steps[-1]['action'] != ('Stop' if decision['prediction'] else 'Forward')):
                raise ValueError(f'Invalid Experiment 2 terminal decision: {source}')
        return decisions

    def _backend_inputs(self, first, seed, rows, order):
        routing = first / f'seed_{seed}/{order}/routing.jsonl'
        decisions = self._saved_decisions(routing, rows, order, allow_pending=True)
        source = first / f'seed_{seed}/audit_requests.jsonl'
        if not source.is_file():
            raise ValueError(f'Missing Experiment 2 Audit requests: {source}')
        requests = [r for r in read_jsonl(source) if r.get('configuration') == order]
        expected_ids = [d['id'] for d in decisions if d['pending_audit']]
        if [r.get('prefix_id') for r in requests] != expected_ids:
            raise ValueError(f'Missing, duplicated or misordered pending requests for seed {seed}, {order}')
        by_id = {r['id']: r for r in rows}
        for item in requests:
            expected = {**request(by_id[item['prefix_id']], self.config['models']['audit']['payload']),
                        'configuration': order, 'seed': seed}
            if any(item.get(k) != v for k, v in expected.items()):
                raise ValueError(f'Pending request ID/payload/seed mismatch: seed {seed}, {order}')
        return decisions, requests, routing, source

    def baselines(self, *, overwrite=False):
        data = self.data_root()
        full = self.compatible('configuration_stage2', ('configuration_stage1',))
        settings = self.config['experiments']['baselines']
        rows = self.rows()
        sources = []
        for seed in self.config['training']['seeds']:
            for order in settings['orders']:
                source = full / f'seed_{seed}/{order}/prefix_decisions.jsonl'
                self._saved_decisions(source, rows, order)
                sources.append((seed, order, source))
        with self.transaction('baselines', ('dataset', 'configuration_stage2'), overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            result = self.executor.run('baseline', {'rows': str(data / 'processed/test_processed.jsonl'),
                        'validation': str(data / 'training/encoder/val.jsonl'),
                        'train': str(data / 'training/autoencoder/train.jsonl')}, out / 'detectors')
            scores = {name: index_scores(read_jsonl(out / f'detectors/{name}_scores.jsonl'), rows)
                      for name in ('sentinel', 'prompt_guard', 'raguard')}
            predictions = {}
            for name in iterable(scores, 'baseline.reports', unit='detectors'):
                predictions[name] = []
                for r in rows:
                    score = scores[name][r['id']]
                    pred = score['prediction'] if name == 'raguard' else int(score['score'] > result[name]['threshold'])
                    predictions[name].append({**r, 'prediction': pred, 'detector': name,
                                              'detector_details': score, 'route': []})
                self.report(out / name, rows, predictions[name])
            combined = [{**r, 'prediction': int(any(predictions[name][i]['prediction'] for name in scores)),
                         'channel_verdicts': {name: predictions[name][i] for name in scores}, 'route': []}
                        for i, r in enumerate(rows)]
            self.report(out / 'combined_or', rows, combined)
            self.report(out / 'no_defense', rows, [{**r, 'prediction': 0, 'route': []} for r in rows])
            for seed, order, source in iterable(sources, 'baseline.ig', unit='seed/configuration pairs'):
                emit(f'baseline.seed_{seed}.{order}', message='Reusing completed Experiment 2 verdicts; no IG inference.', force=True)
                directory = out / f'seed_{seed}/InteractionGuard/{order}'
                decisions = [{**d, 'configuration': order, 'seed': seed} for d in read_jsonl(source)]
                self.report(directory, rows, decisions)
                write_json(directory / 'reuse.json', {'seed': seed, 'configuration': order,
                    'primary': order == self.config['primary_order'], 'new_inference_calls': 0,
                    'source': str(source), 'source_sha256': file_sha256(source),
                    'source_manifest_sha256': file_sha256(full / 'manifest.json'),
                    'timing': 'Reused component estimates from Experiment 2; not a fresh latency repetition.'})
            write_json(out / 'stage_complete.json', {'deterministic_baselines_accuracy_runs': 1,
                         'ig_training_trials': len(self.config['training']['seeds']),
                         'orders': settings['orders'], 'primary_order': self.config['primary_order'],
                         'note': 'No invented independent accuracy repeats for frozen deterministic detectors.'})
            self.aggregate(out, root_reports=True)
        return self.prerequisite('baselines')

    def channel_ablation(self, *, overwrite=False):
        self.data_root()
        settings = self.config['experiments']['channel_ablation']
        with self.transaction('channel_ablation', ('dataset',), overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            rows = self.rows()
            for seed in iterable(self.config['training']['seeds'], 'channel_ablation.seeds', unit='seeds'):
                requests = []
                conditions = [*settings['channels'], 'history']
                for channel in iterable(conditions, f'channel_ablation.seed_{seed}.requests', unit='conditions'):
                    for row in rows:
                        req = request(row, self.config['models']['audit']['payload'],
                                      None if channel == 'history' else channel, settings['prompt_anchoring'])
                        requests.append({**req, 'condition': channel, 'seed': seed})
                answers = self.infer(requests, out / f'seed_{seed}/audit')
                lookup = {(r['condition'], r['prefix_id']): r for r in answers}
                for channel in iterable(conditions + ['channel_or'], f'channel_ablation.seed_{seed}.reports', unit='conditions'):
                    decisions = []
                    for row in rows:
                        used = [lookup[(c, row['id'])] for c in settings['channels']] if channel == 'channel_or' else [lookup[(channel, row['id'])]]
                        decisions.append({**row, 'prediction': int(any(a['prediction'] for a in used)),
                                          'channel_results': used, 'route': [{'layer': 'Audit', 'action': 'Audit'}]})
                    self.report(out / f'seed_{seed}/{channel}', rows, decisions)
                cases = []
                for row in rows:
                    if row.get('attack_type') == 'correlated' and all(row.get('observed_fragments', {}).get(c) for c in ('retrieval', 'tool')):
                        cases.append({'prefix_id': row['id'], 'episode_id': row['original_trace_id'],
                                      'history_positive': bool(lookup[('history', row['id'])]['prediction']),
                                      'all_channel_negative': not any(lookup[(c, row['id'])]['prediction'] for c in settings['channels'])})
                write_jsonl(out / f'seed_{seed}/correlated_cases.jsonl', cases)
            write_json(out / 'stage_complete.json', {'prompt_anchoring': settings['prompt_anchoring'],
                         'limitation': 'Observed two fragments does not prove causal necessity; inspect saved evidence and timing.'})
            self.aggregate(out)
        return self.prerequisite('channel_ablation')

    def backend(self, *, overwrite=False):
        first = self.compatible('configuration_stage1', ('dataset', 'training'))
        settings = self.config['experiments']['backend']
        from research_runner.backend_reuse import validate_reference_candidate, load_reference, publish_reference
        if any(candidate.get('model') == self.config['models']['audit']['model']
               and not candidate.get('reuse_configuration') for candidate in settings['candidates']):
            raise ValueError('The primary Llama backend must reuse Item 2; new inference is forbidden in Item 5')
        reused = [candidate for candidate in settings['candidates'] if candidate.get('reuse_configuration')]
        for candidate in reused:
            validate_reference_candidate(self.config, candidate)
        parents = ('configuration_stage1', 'configuration_stage2') if reused else ('configuration_stage1',)
        rows = self.rows()
        inputs = {}
        # Verify every seed before paying for any backend inference.
        for seed in self.config['training']['seeds']:
            inputs[seed] = self._backend_inputs(first, seed, rows, settings['order'])
        try:
            full = self.compatible('configuration_stage2', ('configuration_stage1',)) if reused else None
        except FileNotFoundError as exc:
            raise ValueError('Missing Experiment 2 configuration_stage2 reference. Complete Item 2 Stage 2; Item 5 will not rerun Llama.') from exc
        references = {seed: load_reference(self, full, first, seed, rows, inputs[seed][0], inputs[seed][1])
                      for seed in self.config['training']['seeds']} if reused else {}
        preflight = None
        if isinstance(self.executor, ProcessExecutor):
            preflight = self.executor.preflight_audit(settings['candidates'])
        with self.transaction('backend', parents, overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            if preflight is not None:
                write_json(out / 'runtime_preflight.json', preflight)
            for seed in iterable(self.config['training']['seeds'], 'backend.seeds', unit='seeds'):
                staged, requests, routing, source = self._backend_inputs(first, seed, rows, settings['order'])
                expected = fingerprint([(r['prefix_id'], r['request_id']) for r in requests])
                for candidate in iterable(settings['candidates'], f'backend.seed_{seed}', unit='backends'):
                    emit(f'backend.seed_{seed}.{candidate["name"]}', message=f'{len(requests)} pending requests; upstream routing fixed.', force=True)
                    directory = out / f'seed_{seed}/{candidate["name"]}'
                    if candidate.get('reuse_configuration'):
                        emit(f'backend.seed_{seed}.{candidate["name"]}', message='Reusing exact Item 2 reference; zero new inference and inherited timing.', force=True)
                        publish_reference(references[seed], directory)
                    else:
                        answers = self.infer(requests, directory / 'audit', candidate)
                        lookup = {r['prefix_id']: r for r in answers}
                        decisions = deepcopy(staged)
                        for d in decisions:
                            if d['pending_audit']:
                                answer = lookup[d['id']]
                                resolve_pending(d, answer)
                            d.update(configuration=settings['order'], seed=seed)
                        self.report(directory, rows, decisions)
                    write_json(directory / 'routing_identity.json', {'pending_set_sha256': expected,
                               'seed': seed, 'backend': candidate, 'same_upstream_routing': True,
                               'configuration': settings['order'], 'ordered_prefix_ids': [r['prefix_id'] for r in requests],
                               'routing_sha256': file_sha256(routing), 'requests_sha256': file_sha256(source)})
            write_json(out / 'comparison_protocol.json', {
                candidate['name']: {'reference_reused_from_item_2': bool(candidate.get('reuse_configuration')),
                                    'new_inference_trials': 0 if candidate.get('reuse_configuration') else len(self.config['training']['seeds']),
                                    'latency_source': 'inherited_item_2' if candidate.get('reuse_configuration') else 'new_item_5_inference'}
                for candidate in settings['candidates']})
            write_json(out / 'stage_complete.json', {'source_stage1': str(first),
                       'configuration': settings['order'], 'primary_order': self.config['primary_order']})
            self.aggregate(out)
            summary = read_json(out / 'summary.json')
            for candidate in reused:
                summary[candidate['name']].update(
                    variation='inherited_item_2_seed_results_not_new_inference_trials',
                    reference_reused_from_item_2=True, new_inference_trials=0,
                    latency_source='inherited_item_2')
            write_json(out / 'summary.json', summary)
        return self.prerequisite('backend')

    def threshold(self, *, overwrite=False):
        from research_runner.threshold_reuse import ThresholdReference, publish_threshold_reference, finish_threshold_summary
        first = self.compatible('configuration_stage1', ('dataset', 'training'))
        settings = self.config['experiments']['threshold']
        rows = self.rows()
        validation = read_jsonl(self.data_root() / 'training/encoder/val.jsonl')
        screen = index_scores(read_jsonl(first / 'scores/screen_scores.jsonl'), rows)
        screen_threshold = read_json(first / 'scores/screen_calibration.json')['threshold']
        score_sets = {seed: (
            index_scores(read_jsonl(first / f'scores/gate_scores_seed{seed}.jsonl'), rows),
            index_scores(read_jsonl(first / f'scores/gate_validation_seed{seed}.jsonl'), validation))
            for seed in self.config['training']['seeds']}
        score_sources = {'stage1_manifest_sha256': file_sha256(first / 'manifest.json'),
                         'screen_test_sha256': file_sha256(first / 'scores/screen_scores.jsonl'),
                         'screen_calibration_sha256': file_sha256(first / 'scores/screen_calibration.json')}
        reference = ThresholdReference(self, first)
        parents = ('configuration_stage1', 'configuration_stage2') if reference.full else ('configuration_stage1',)
        with self.transaction('threshold', parents, overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            for seed in iterable(self.config['training']['seeds'], 'threshold.seeds', unit='seeds'):
                gate, val = score_sets[seed]
                gates = [(f'recall_{floor}', select_gate_threshold([r['label'] for r in validation],
                         [val[r['id']]['score'] for r in validation], floor)) for floor in settings['recall_floors']]
                if settings['include_validation_max_f1']:
                    gates.insert(0, ('validation_max_f1', calibrate_gate([r['label'] for r in validation],
                                   [val[r['id']]['score'] for r in validation], 0.)))
                for name, calibration in iterable(gates, f'threshold.seed_{seed}', unit='operating points'):
                    emit(f'threshold.seed_{seed}.{name}', message=f"Gate threshold={calibration['threshold']}; other settings fixed.", force=True)
                    directory = out / f'seed_{seed}/{name}'
                    shared = {'seed': seed, 'operating_point': name, 'gate': calibration,
                              'sweep_selection_policy': settings.get('selection_policy', 'max_threshold_under_recall_floor'),
                              'screen_threshold': screen_threshold, 'sample_ids': [r['id'] for r in rows],
                              'selection_source': 'validation_only', 'other_models_and_policies_fixed': True,
                              'score_sources': {**score_sources,
                                  'gate_test_sha256': file_sha256(first / f'scores/gate_scores_seed{seed}.jsonl'),
                                  'gate_validation_sha256': file_sha256(first / f'scores/gate_validation_seed{seed}.jsonl')}}
                    shared['paired_workload_sha256'] = fingerprint(shared)
                    write_json(directory / 'operating_point.json', shared)
                    for order in iterable(settings['orders'], f'threshold.seed_{seed}.{name}.orders', unit='configurations'):
                        emit(f'threshold.seed_{seed}.{name}.{order}', message='Same threshold and ordered test IDs; only cascade order differs.', force=True)
                        decisions = route(rows, gate, screen, order.split('-'), calibration['threshold'], screen_threshold)
                        pending = {d['id'] for d in decisions if d['pending_audit']}
                        requests = [{**request(r, self.config['models']['audit']['payload']),
                                     'configuration': order, 'seed': seed, 'operating_point': name}
                                    for r in rows if r['id'] in pending]
                        condition = directory / order
                        bundle, reason = reference.lookup(seed, order, rows, decisions, requests, calibration)
                        if bundle is not None:
                            emit(f'threshold.seed_{seed}.{name}.{order}', message='Reusing verified Item 2 verdicts; zero new Audit requests, inherited latency.', force=True)
                            publish_threshold_reference(bundle, condition)
                        else:
                            emit(f'threshold.seed_{seed}.{name}.{order}', message=f'Fresh inference. Reuse: {reason}', force=True)
                            lookup = {a['prefix_id']: a for a in self.infer(requests, condition / 'audit')}
                            for d in decisions:
                                if d['pending_audit']:
                                    resolve_pending(d, lookup[d['id']])
                                d.update(configuration=order, seed=seed)
                            self.report(condition, rows, decisions)
                        execution = {'seed': seed, 'operating_point': name, 'configuration': order,
                                     'result_source': 'inherited_item_2' if bundle is not None else 'new_item_6_inference',
                                     'latency_source': 'inherited_item_2' if bundle is not None else 'new_item_6_audit_component',
                                     'new_audit_requests': read_json(condition / 'audit/timing.json')['submitted_requests'],
                                     'reuse_fallback_reason': reason}
                        write_json(condition / 'execution.json', execution)
                        write_json(condition / 'operating_point.json', {**shared, 'configuration': order,
                                   'primary': order == self.config['primary_order']})
            write_json(out / 'stage_complete.json', {'variable': 'validation-derived Gate threshold',
                       'selection_policy': settings.get('selection_policy', 'max_threshold_under_recall_floor'),
                       'orders': settings['orders'], 'primary_order': self.config['primary_order'],
                       'paired_workloads_across_configurations': True})
            self.aggregate(out)
            finish_threshold_summary(out)
        return self.prerequisite('threshold')

    def benign(self, *, overwrite=False):
        import random
        first = self.compatible('configuration_stage1', ('dataset', 'training'))
        training = self.compatible('training', ('dataset',))
        checkpoints = read_json(training / 'checkpoints.json')
        settings = self.config['experiments']['benign']
        orders = settings['orders']
        if settings['profile'] != 'main':
            raise ValueError('Only the frozen main benign-workload profile is supported; no silent threshold retuning.')
        with self.transaction('benign', ('configuration_stage1',), overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            all_rows = self.rows()
            screen = index_scores(read_jsonl(first / 'scores/screen_scores.jsonl'), all_rows)
            screen_threshold = read_json(first / 'scores/screen_calibration.json')['threshold']
            for trial, seed in enumerate(iterable(self.config['training']['seeds'], 'benign.seeds', unit='seeds')):
                gate = index_scores(read_jsonl(first / f'scores/gate_scores_seed{seed}.jsonl'), all_rows)
                for ratio in iterable(settings['ratios'], f'benign.seed_{seed}', unit='ratios'):
                    emit(f'benign.seed_{seed}.{ratio}', message=f'Benign ratio={ratio:.0%}; thresholds fixed.', force=True)
                    rng = random.Random(settings['sampling_seed'] + trial * 10000 + round(ratio * 100))
                    negatives = [r for r in all_rows if r['label'] == 0]
                    positive_groups = defaultdict(list)
                    for r in all_rows:
                        if r['label'] == 1:
                            positive_groups[r['attack_type']].append(r)
                    n = settings['total_size']
                    n_benign = int(n * ratio)
                    if n_benign > len(negatives):
                        raise ValueError('Not enough benign prefixes for configured workload; no silent replacement.')
                    positives = []
                    for group in sorted(positive_groups):
                        shuffled = list(positive_groups[group]); rng.shuffle(shuffled)
                        positive_groups[group] = shuffled
                    while len(positives) < n-n_benign:
                        progress = False
                        for group in sorted(positive_groups):
                            if positive_groups[group] and len(positives) < n-n_benign:
                                positives.append(positive_groups[group].pop()); progress = True
                        if not progress:
                            raise ValueError('Not enough malicious prefixes for workload.')
                    rows = rng.sample(negatives, n_benign) + positives
                    rng.shuffle(rows)
                    directory = out / f'seed_{seed}/benign_{round(ratio*100)}'
                    workload = {'requested_benign_ratio': ratio, 'actual_benign_ratio': n_benign/n,
                           'sample_ids': [r['id'] for r in rows], 'sampling_seed': settings['sampling_seed'] + trial*10000 + round(ratio*100),
                           'gate_threshold': checkpoints[str(seed)]['gate_threshold'], 'screen_threshold': screen_threshold,
                           'profile': 'main', 'sampling_unit': 'prefix_without_replacement_stratified_attack_categories',
                           'asr_note': 'Partial-prefix workload does not support whole-episode ASR/false-blocking; use Experiment 3.'}
                    workload['paired_workload_sha256'] = fingerprint(workload)
                    write_json(directory / 'workload.json', workload)
                    for order in iterable(orders, f'benign.seed_{seed}.{ratio}.orders', unit='configurations'):
                        emit(f'benign.seed_{seed}.{ratio}.{order}', message='Same sampled workload; no threshold retuning.', force=True)
                        decisions = route(rows, gate, screen, order.split('-'), checkpoints[str(seed)]['gate_threshold'], screen_threshold)
                        pending = {d['id'] for d in decisions if d['pending_audit']}
                        requests = [request(r, self.config['models']['audit']['payload']) for r in rows if r['id'] in pending]
                        condition = directory / order
                        lookup = {a['prefix_id']: a for a in self.infer(requests, condition / 'audit')}
                        for d in decisions:
                            if d['pending_audit']:
                                resolve_pending(d, lookup[d['id']])
                        self.report(condition, rows, decisions, episode_level=False)
                        benign_front_pass = sum(r['label'] == 0 and d['route'][0] == {'layer': 'Gate', 'action': 'Forward'}
                                                for r, d in zip(rows, decisions))
                        write_json(condition / 'workload.json', {**workload, 'configuration': order,
                                   'benign_front_gate_forward_count': benign_front_pass,
                                   'benign_front_gate_forward_ratio': benign_front_pass / n_benign if n_benign else 0.})
            write_json(out / 'stage_complete.json', {'variable': 'benign prefix ratio; thresholds fixed within each seed',
                       'orders': orders, 'paired_workloads_across_configurations': True})
            self.aggregate(out)
        return self.prerequisite('benign')

    def configuration_stage2(self, *, overwrite=False):
        first = self.compatible('configuration_stage1', ('dataset', 'training'))
        settings = self.config['experiments']['configuration']
        with self.transaction('configuration_stage2', ('configuration_stage1',), overwrite, settings['resume']) as out:
            if (out / 'stage_complete.json').exists():
                return out
            rows = self.rows()
            for seed in iterable(self.config['training']['seeds'], 'stage2.seeds', unit='seeds'):
                requests = read_jsonl(first / f'seed_{seed}/audit_requests.jsonl')
                answers = self.infer(requests, out / f'seed_{seed}/audit')
                lookup = {(a['configuration'], a['prefix_id']): a for a in answers}
                for order in iterable(settings['orders'], f'stage2.seed_{seed}.reports', unit='configurations'):
                    decisions = read_jsonl(first / f'seed_{seed}/{order}/routing.jsonl')
                    for d in decisions:
                        if d['pending_audit']:
                            a = lookup[(order, d['id'])]
                            resolve_pending(d, a)
                    self.report(out / f'seed_{seed}/{order}', rows, decisions)
            write_json(out / 'stage_complete.json', {'stage': 2, 'stage1_manifest_sha256': file_sha256(first / 'manifest.json')})
            self.aggregate(out)
        return self.prerequisite('configuration_stage2')

    def aggregate(self, out, *, root_reports=False):
        emit('aggregate', message=f'Summarizing trials and writing tables/figures: {out}', force=True)
        groups = defaultdict(list)
        for path in sorted(Path(out).glob('seed_*/**/metrics.json')):
            group = '/'.join(path.relative_to(out).parts[1:-1])
            groups[group].append(path)
        if root_reports:
            for path in sorted(Path(out).glob('*/metrics.json')):
                groups[path.parent.name].append(path)
        summary = {}
        for group, paths in groups.items():
            values = [read_json(p) for p in paths]
            deterministic = root_reports and '/' not in group
            summary[group] = {'trials': len(values),
                              'variation': 'frozen_baseline_single_accuracy_evaluation_not_independent_repeats' if deterministic else 'paired_trials; see effective_config for training/sampling/inference sources',
                              'metrics': {key: {'mean': statistics.mean(v[key] for v in values),
                                                'std': statistics.stdev(v[key] for v in values) if len(values) > 1 else 0.}
                                          for key in ('precision', 'recall', 'f1', 'fpr', 'fnr', 'estimated_pipeline_seconds_per_sample')},
                              'traffic': {layer: {'mean': statistics.mean(v['processed_ratios'][layer] for v in values),
                                                  'std': statistics.stdev(v['processed_ratios'][layer] for v in values) if len(values) > 1 else 0.}
                                          for layer in ('Gate', 'Screen', 'Audit')}}
            reports = [read_json(p.parent / 'asr.json') for p in paths if (p.parent / 'asr.json').exists()]
            if reports:
                def stats(items):
                    defined = [v for v in items if v is not None]
                    return {'defined_trials': len(defined), 'mean': statistics.mean(defined) if defined else None,
                            'std': statistics.stdev(defined) if len(defined) > 1 else 0. if defined else None}
                summary[group]['asr'] = {
                    'pooled': stats([a['global']['asr'] for a in reports]),
                    'macro': stats([a['macro_asr'] for a in reports]),
                    'benign_episode_false_block_rate': stats([a['benign']['false_block_rate'] for a in reports]),
                    'categories': {category: {
                        'asr': stats([a['per_attack_type'][category]['asr'] for a in reports]),
                        'attempts_per_trial': [a['per_attack_type'][category]['total'] for a in reports],
                        'baseline_successes_per_trial': [a['per_attack_type'][category]['baseline_successful'] for a in reports],
                        'surviving_successes_per_trial': [a['per_attack_type'][category]['success'] for a in reports]}
                        for category in reports[0]['per_attack_type']}}
        write_json(Path(out) / 'summary.json', summary)
        if self.config['experiments']['timing'].get('generate_figures', True):
            from research_runner.plots import render
            write_json(Path(out) / 'figures.json', render(summary, out))
        with (Path(out) / 'summary.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['condition', 'trials', 'metric', 'mean', 'std'])
            for name, group in summary.items():
                for metric, value in group['metrics'].items():
                    writer.writerow([name, group['trials'], metric, value['mean'], value['std']])
                for layer, value in group['traffic'].items():
                    writer.writerow([name, group['trials'], f'processed_ratio.{layer}', value['mean'], value['std']])
                for metric, value in group.get('asr', {}).items():
                    if metric == 'categories':
                        for category, record in value.items():
                            item = record['asr']
                            writer.writerow([name, group['trials'], f'asr.category.{category}', item['mean'], item['std']])
                    else:
                        writer.writerow([name, group['trials'], f'asr.{metric}', value['mean'], value['std']])
