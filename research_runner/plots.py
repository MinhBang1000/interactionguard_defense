"""Render readable paper figures from completed aggregates, never model calls."""
from pathlib import Path


def render(summary, directory, *, traffic_layers=('Screen', 'Audit')):
    if not summary:
        return {'created': [], 'note': 'No multi-trial metrics to plot.'}
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return {'created': [], 'note': 'Install matplotlib in the runner environment to render figures; JSON/CSV are complete.'}
    if all(len(label.split('/')) == 2 and label.split('/')[0].startswith('benign_') for label in summary):
        return _paired_benign(summary, directory, plt)
    if all(len(label.split('/')) == 2 and label.split('/')[0].startswith(('recall_', 'validation_max_f1')) for label in summary):
        return _paired_threshold(summary, directory, plt)
    labels = list(summary)
    values = [summary[label] for label in labels]
    positions = list(range(len(labels)))
    fig, left = plt.subplots(figsize=(9, 4.8))
    right = left.twinx()
    left.errorbar(positions, [v['metrics']['f1']['mean'] * 100 for v in values],
                  yerr=[v['metrics']['f1']['std'] * 100 for v in values],
                  color='#222222', linestyle='--', marker='^', label='F1 score', capsize=3)
    sweep = all(label.startswith(('recall_', 'validation_max_f1', 'benign_')) for label in labels)
    if sweep:
        styles = {'Screen': ('#0077aa', 'o'), 'Gate': ('#886633', 'v'), 'Audit': ('#009977', 's')}
        for layer in traffic_layers:
            color, marker = styles[layer]
            left.errorbar(positions, [v['traffic'][layer]['mean'] * 100 for v in values],
                          yerr=[v['traffic'][layer]['std'] * 100 for v in values],
                          color=color, marker=marker, capsize=3, label=f'Samples processed by {layer}')
    latency = [v['metrics']['estimated_pipeline_seconds_per_sample']['mean'] for v in values]
    right.errorbar(positions, latency,
                   yerr=[v['metrics']['estimated_pipeline_seconds_per_sample']['std'] for v in values],
                   color='#cc5500', marker='D', label='Estimated pipeline time/sample', capsize=3)
    right.set_ylim(0, max(latency, default=0) * 1.35 or 1.)
    left.set_ylim(0, 105)
    left.set_ylabel('Processed sample ratio / F1 score (%)' if sweep else 'F1 score (%)')
    right.set_ylabel('Component-based pipeline estimate (s/sample)')
    left.set_xticks(positions, [label.replace('InteractionGuard/', 'IG: ') for label in labels], rotation=20, ha='right')
    left.grid(axis='y', alpha=.2)
    lines, names = left.get_legend_handles_labels()
    lines2, names2 = right.get_legend_handles_labels()
    fig.legend(lines + lines2, names + names2, loc='upper center', bbox_to_anchor=(.5, 1.02),
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .87))
    # No chart title: the paper supplies the LaTeX caption.
    output = Path(directory) / 'comparison.png'
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)
    return {'created': [output.name], 'note': 'Mean +/- sample standard deviation across paired trials; see manifest for variation sources.'}


def _paired_threshold(summary, directory, plt):
    """Keep the complete traffic chart per order, plus an order comparison."""
    directory = Path(directory)
    orders = sorted({key.split('/')[1] for key in summary})
    points = list(dict.fromkeys(key.split('/')[0] for key in summary))
    if len(summary) != len(orders) * len(points):
        raise ValueError('Paired threshold plot requires every configuration at every operating point')
    created = []
    for order in orders:
        child = directory / order
        child.mkdir(parents=True, exist_ok=True)
        result = render({point: summary[f'{point}/{order}'] for point in points}, child,
                        traffic_layers=('Screen', 'Gate', 'Audit'))
        created.extend(f'{order}/{name}' for name in result['created'])
    fig, left = plt.subplots(figsize=(10, 5))
    right = left.twinx()
    positions = list(range(len(points)))
    upper = []
    for index, order in enumerate(orders):
        values = [summary[f'{point}/{order}'] for point in points]
        color = ['#0077aa', '#cc5500', '#009977', '#222222'][index % 4]
        left.errorbar(positions, [v['metrics']['f1']['mean'] * 100 for v in values],
                      yerr=[v['metrics']['f1']['std'] * 100 for v in values],
                      color=color, linestyle='--', marker='^', capsize=3, label=f'{order}: F1')
        times = [v['metrics']['estimated_pipeline_seconds_per_sample']['mean'] for v in values]
        deviations = [v['metrics']['estimated_pipeline_seconds_per_sample']['std'] for v in values]
        right.errorbar(positions, times, yerr=deviations, color=color, marker='D', capsize=3,
                       label=f'{order}: estimated time')
        upper.extend(mean + std for mean, std in zip(times, deviations))
    left.set_ylim(0, 105)
    right.set_ylim(0, max(upper, default=0.) * 1.35 or 1.)
    left.set_xticks(positions, points, rotation=20, ha='right')
    left.set_xlabel('Validation-derived Gate operating point (threshold is seed-specific)')
    left.set_ylabel('F1 score (%)')
    right.set_ylabel('Component-based pipeline estimate (s/sample)')
    left.grid(axis='y', alpha=.2)
    handles, names = left.get_legend_handles_labels()
    handles2, names2 = right.get_legend_handles_labels()
    fig.legend(handles + handles2, names + names2, loc='upper center', bbox_to_anchor=(.5, 1.02),
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .85))
    output = directory / 'comparison.png'
    fig.savefig(output, dpi=220, bbox_inches='tight')
    plt.close(fig)
    created.insert(0, output.name)
    return {'created': created, 'note': 'Paired orders on identical test IDs and per-seed validation thresholds; mean +/- sample standard deviation.'}


def _paired_benign(summary, directory, plt):
    """Compare orders on identical ratios, retaining each order's traffic chart."""
    directory = Path(directory)
    configurations = sorted({label.split('/')[1] for label in summary})
    ratios = sorted({int(label.split('/')[0].split('_')[1]) for label in summary})
    if len(summary) != len(configurations) * len(ratios):
        raise ValueError('Paired benign plot requires every configuration at every ratio')
    created = []
    for order in configurations:
        child = directory / order
        child.mkdir(parents=True, exist_ok=True)
        subset = {f'benign_{ratio}': summary[f'benign_{ratio}/{order}'] for ratio in ratios}
        files = render(subset, child)
        created.extend(f'{order}/{name}' for name in files['created'])
    fig, left = plt.subplots(figsize=(9, 4.8))
    right = left.twinx()
    all_latency = []
    colors = ['#0077aa', '#cc5500', '#009977', '#222222']
    for index, order in enumerate(configurations):
        values = [summary[f'benign_{ratio}/{order}'] for ratio in ratios]
        color = colors[index % len(colors)]
        left.errorbar(ratios, [v['metrics']['f1']['mean'] * 100 for v in values],
                      yerr=[v['metrics']['f1']['std'] * 100 for v in values],
                      color=color, linestyle='--', marker='^', capsize=3, label=f'{order}: F1')
        times = [v['metrics']['estimated_pipeline_seconds_per_sample']['mean'] for v in values]
        deviations = [v['metrics']['estimated_pipeline_seconds_per_sample']['std'] for v in values]
        right.errorbar(ratios, times, yerr=deviations, color=color, marker='D', capsize=3,
                       label=f'{order}: estimated time')
        all_latency.extend(mean + std for mean, std in zip(times, deviations))
    left.set_ylim(0, 105)
    right.set_ylim(0, max(all_latency, default=0.) * 1.35 or 1.)
    left.set_xticks(ratios)
    left.set_xlabel('Benign-labeled prefixes (%)')
    left.set_ylabel('F1 score (%)')
    right.set_ylabel('Component-based pipeline estimate (s/sample)')
    left.grid(axis='y', alpha=.2)
    lines, labels = left.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    fig.legend(lines + lines2, labels + labels2, loc='upper center', bbox_to_anchor=(.5, 1.02),
               ncol=2, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, .85))
    path = directory / 'comparison.png'
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)
    created.insert(0, path.name)
    return {'created': created, 'note': 'Paired orders on identical sampled prefix IDs; mean +/- sample standard deviation. No episode ASR.'}
