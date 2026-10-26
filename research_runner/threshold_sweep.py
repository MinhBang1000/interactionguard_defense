"""Validation-only operating-point selection for Item 6, not model calibration."""
from collections import Counter, defaultdict

from research_runner.scoring import _finite, _score_metrics, _validation


POLICY = 'max_threshold_under_recall_floor'


def validate_settings(settings):
    if settings.get('selection_policy', POLICY) != POLICY:
        raise ValueError(f'Gate sweep selection_policy must be {POLICY}')
    if 'recall_floors' not in settings:
        if settings.get('reuse_configuration_recall_floors'):
            raise ValueError('Reuse recall floors require explicit recall_floors')
        return
    floors = settings['recall_floors']
    if not isinstance(floors, list) or not floors:
        raise ValueError('Gate sweep recall_floors must be a nonempty list')
    values = [_finite(value, 'recall floor', True) for value in floors]
    if any(value <= 0 for value in values) or values != sorted(set(values)):
        raise ValueError('Gate sweep recall_floors must be distinct, increasing and in (0, 1]')
    reuse = settings.get('reuse_configuration_recall_floors', [])
    if not isinstance(reuse, list):
        raise ValueError('reuse_configuration_recall_floors must be a list')
    selected = [_finite(value, 'reuse recall floor', True) for value in reuse]
    if selected != sorted(set(selected)) or not set(selected) <= set(values):
        raise ValueError('Reuse recall floors must be distinct increasing members of recall_floors')


def select_gate_threshold(labels, scores, recall_floor):
    """Highest empirical score cutoff satisfying recall with strict score > cutoff.

    Among empirical cutoffs, this minimizes forwarded anomaly traffic subject to
    the recall floor. Tied scores cannot be split to manufacture target recall.
    The separate max-F1 anchor still uses the original calibration function.
    """
    labels, scores = _validation(labels, scores)
    recall_floor = _finite(recall_floor, 'recall floor', True)
    if recall_floor <= 0:
        raise ValueError('Gate sweep recall floor must be in (0, 1]')
    groups = defaultdict(Counter)
    for score, label in zip(scores, labels):
        groups[score][label] += 1
    positives = sum(labels)
    threshold = min(scores) - max(1e-12, abs(min(scores)) * 1e-6)
    tp = positives
    for score in sorted(groups):
        tp -= groups[score][1]
        if tp / positives < recall_floor:
            break
        threshold = score
    measured = _score_metrics(labels, scores, threshold)
    return {'objective': POLICY, 'recall_floor': recall_floor, 'threshold': threshold,
            'metrics': measured, 'candidate_thresholds': len(groups) + 1,
            'comparison': 'strict_greater_than',
            'validation_escalation_ratio': (measured['tp'] + measured['fp']) / len(labels),
            'selection_population': 'prepared_noncorrelated_validation_prefixes'}
