"""Noninteractive worker entry point. Import model dependencies only in adapters."""
import json
from pathlib import Path
import sys

from research_runner.progress import emit, iterable, scope


def execute(operation, config, job, output, backend=None):
    from research_runner.artifacts import read_jsonl, write_json, write_jsonl
    from research_runner.scoring import calibrate_l2
    if backend is None:
        from research_runner.adapters import RealBackend
        backend = RealBackend(config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(job['rows']) if job.get('rows') else []
    emit(f'worker.{operation}', message=f"Seed={job.get('seed', 'shared')}; {len(rows)} input prefixes; output={output}", force=True)
    if operation == 'train':
        result = backend.train(job['seed'], output / 'model')
    elif operation == 'score':
        emit('score.l2.test', message=f'{len(rows)} test prefixes.', force=True)
        l2 = backend.score_detector(rows, config['models']['l2'])
        validation = read_jsonl(job['validation'])
        emit('score.l2.validation', message=f'{len(validation)} validation prefixes; then threshold calibration.', force=True)
        val_scores = backend.score_detector(validation, config['models']['l2'])
        calibration = calibrate_l2([r['label'] for r in validation], [r['score'] for r in val_scores],
                                   config['training']['calibration']['l2_fpr_budget'])
        write_jsonl(output / 'l2_scores.jsonl', l2)
        write_jsonl(output / 'l2_validation_scores.jsonl', val_scores)
        write_json(output / 'l2_calibration.json', calibration)
        checkpoints = job['checkpoints']
        result = {'l2_calibration': calibration, 'seeds': {}}
        for seed, checkpoint in iterable(checkpoints.items(), 'score.l1.seeds', unit='seeds'):
            emit(f'score.l1.seed_{seed}', message='Scoring test and validation prefixes.', force=True)
            scores = backend.score_l1(rows, checkpoint)
            write_jsonl(output / f'l1_scores_seed{seed}.jsonl', scores)
            write_jsonl(output / f'l1_validation_seed{seed}.jsonl', backend.score_l1(validation, checkpoint))
            result['seeds'][seed] = {'checkpoint': checkpoint, 'scores_file': f'l1_scores_seed{seed}.jsonl'}
    elif operation == 'infer':
        requests = read_jsonl(job['requests'])
        responses = backend.infer_l3(requests, job.get('backend'))
        write_jsonl(output / 'responses.jsonl', responses)
        result = {'requests': len(requests), 'responses': len(responses)}
    elif operation == 'baseline':
        train = read_jsonl(job['train'])
        validation = read_jsonl(job['validation'])
        result = {}
        settings = config['experiments']['baselines']
        for name, channel in iterable((('sentinel', 'prompt'), ('prompt_guard', 'tool')), 'baseline.models', unit='detectors'):
            emit(f'baseline.{name}', message=f'Channel={channel}; scoring test and validation, then calibration.', force=True)
            values = backend.score_detector(rows, settings[name], channels=[channel])
            val = backend.score_detector(validation, settings[name], channels=[channel])
            calibration = calibrate_l2([r['label'] for r in validation], [r['score'] for r in val], settings['fpr_budget'])
            write_jsonl(output / f'{name}_scores.jsonl', values)
            write_jsonl(output / f'{name}_validation.jsonl', val)
            result[name] = calibration
        rag = backend.score_rag(rows, train, settings['raguard'])
        write_jsonl(output / 'raguard_scores.jsonl', rag)
        result['raguard'] = {'calibration': 'prepared_benign_noncorrelated_training_documents',
                            'alpha': settings['raguard']['alpha'], 'training_ids': [r['id'] for r in train]}
    else:
        raise ValueError(f'Unknown worker operation: {operation}')
    close = getattr(backend, 'close', None)
    if callable(close):
        close()
    write_json(output / 'worker_result.json', result)
    emit(f'worker.{operation}', status='completed', message='Worker results saved.', force=True)
    return result


def main():
    path = Path(sys.argv[1])
    spec = json.loads(path.read_text())
    with scope(path.parent / 'progress', f"worker.{spec['operation']}"):
        execute(spec['operation'], spec['config'], spec['job'], path.parent)


if __name__ == '__main__':
    main()
