"""Functional identities are stable even when a cascade order changes."""

FUNCTIONAL_NAMES = frozenset({'Screen', 'Gate', 'Audit'})


def paper_order(order, config):
    """Translate labels for a paper export, never mutate runtime identities."""
    names = order.split('-')
    if not names or not set(names) <= FUNCTIONAL_NAMES or len(names) != len(set(names)):
        raise ValueError(f'Invalid functional order: {order}')
    return '-'.join(config['naming']['paper'][name] for name in names)
