"""Small non-parameterized menu for the InteractionGuard campaign."""
import argparse
from pathlib import Path

from research_runner.artifacts import ExistingResultsError
from research_runner.config import load_config
from research_runner.engine import ResearchRunner

MENU = {
    '0': 'Full Dataset Preparation',
    '1': 'Train and Calibrate Encoder + Gate',
    '2': 'Layer Configuration Comparison',
    '3': 'Channel-Specific Baseline and ASR Comparison',
    '4': 'Same-Audit Channel vs Full-History Comparison',
    '5': 'Audit Backend Comparison',
    '6': 'Gate Threshold Sweep',
    '7': 'Benign-Ratio Sweep',
    '8': 'Auditor-Targeted Case Stress Test',
    '9': 'Backup Snapshot',
}


def run_selection(runner, choice, *, input_fn=input, print_fn=print):
    item = int(choice)
    print_fn(f'Running Experiment {choice}: {MENU[choice]}')
    print_fn('Parameters are read from the single configuration file; no runtime parameter prompts.')
    try:
        result = runner.run(item)
    except ExistingResultsError as exc:
        print_fn(str(exc))
        if input_fn('Existing results found. Overwrite this experiment\'s results? [y/N] ').strip().lower() not in {'y', 'yes'}:
            cancel = getattr(runner, 'cancel', None)
            if callable(cancel):
                cancel(item)
            print_fn('Cancelled. Existing results are unchanged.')
            return None
        result = runner.run(item, overwrite=True)
    print_fn(f'Completed. Output: {result}' if result is not None else 'Completed.')
    return result


def main(argv=None, *, input_fn=input, print_fn=print, runner_factory=ResearchRunner):
    parser = argparse.ArgumentParser(description='InteractionGuard research runner')
    parser.add_argument('--config', default='config/experiments.yaml')
    parser.add_argument('--run', choices=MENU, help='Select a menu item; parameters come only from YAML')
    args = parser.parse_args(argv)
    config = load_config(args.config)
    runner = runner_factory(config)
    print_fn('InteractionGuard | Offline Research Runner')
    print_fn(f'Configuration: {Path(args.config).resolve()}')
    while True:
        if args.run is None:
            for key, label in MENU.items():
                print_fn(f'{key}. {label}')
            print_fn('Q. Exit')
            choice = input_fn('Select experiment: ').strip().upper()
        else:
            choice = args.run
        if choice == 'Q':
            return 0
        if choice not in MENU:
            print_fn('Invalid selection. Choose 0-9 or Q.')
            continue
        try:
            run_selection(runner, choice, input_fn=input_fn, print_fn=print_fn)
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            print_fn(f'Experiment failed: {exc}')
            if args.run is not None:
                return 1
        if args.run is not None:
            return 0
