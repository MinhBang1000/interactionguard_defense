"""Baseline adapters; episode semantics are implemented only in eval.asr."""
from pathlib import Path
from data.data_preprocess import ROOT, write_json
from eval.asr import evaluate_files, original_outcome


def is_successful_attack(trace):
    return original_outcome(trace)["success"]


def ground_truth_success(sample):
    from eval.asr import load_judges
    return original_outcome(sample, load_judges(ROOT / "eval/prompt_success_judgements.jsonl"))["success"]


def evaluate_predictions(prediction_file, attack_types=None, save_outputs=True,
                         attack_defense_type="rag", *, allow_unverified=False):
    report = evaluate_files(ROOT / "data/splits/test.jsonl", predictions=Path(prediction_file),
                            processed=ROOT / "data/processed/test_processed.jsonl",
                            judges=ROOT / "eval/prompt_success_judgements.jsonl",
                            allow_unverified=allow_unverified)
    if save_outputs:
        write_json(ROOT / "eval" / f"{attack_defense_type}_attack_eval_report.json", report["per_attack_type"])
        write_json(ROOT / "eval" / f"{attack_defense_type}_attack_eval_details.json", report)
    return report


def run_eval_asr_for_baseline(paths=None, *, allow_unverified=False):
    from eval.eval_asr_within_defense import eval_with_baseline
    if paths is None:
        return eval_with_baseline(allow_unverified=allow_unverified)
    return {name: evaluate_predictions(path, attack_defense_type=name, allow_unverified=allow_unverified)
            for name, path in paths.items()}


if __name__ == "__main__":
    run_eval_asr_for_baseline()
