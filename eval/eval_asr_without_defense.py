"""Compatibility API; all ASR semantics live in eval.asr."""
from data.data_preprocess import ROOT, write_json
from eval.asr import evaluate_files, original_outcome


def is_successful_attack(trace):
    return original_outcome(trace)["success"]


def filter_attack_type(data_path="test.jsonl", attack_types=None, save_outputs=True):
    report = evaluate_files(ROOT / "data/splits" / data_path,
                            judges=ROOT / "eval/prompt_success_judgements.jsonl")
    stats = report["per_attack_type"]
    if attack_types is not None:
        stats = {k: v for k, v in stats.items() if k in attack_types}
    if save_outputs:
        write_json(ROOT / "eval/asr_offline_no_defense.json", report)
        write_json(ROOT / "eval/eval_asr_without_defense.json", stats)
    for kind, row in stats.items():
        print(f"{kind}: attempts={row['total']} success={row['success']} unknown={row['unknown']} ASR={row['asr']}")
    return stats


if __name__ == "__main__":
    filter_attack_type()
