"""Compatibility adapters; every ASR decision delegates to eval.asr.

Old predictions without dataset provenance are rejected by default. Episode-only
blocks without detection timing cannot demonstrate timely prevention.
"""
from data.data_preprocess import ROOT, write_json
from eval.asr import evaluate_files


def eval_with_defense(data_path="test_predicted.jsonl", save_name="eval_asr_within_defense.json",
                      create_if_missing=True, source_processed_path="test_processed.jsonl",
                      source_predict_path="mode_11_results.json", *, allow_unverified=False):
    # Always read source predictions, never a stale aggregated episode cache.
    report = evaluate_files(ROOT / "data/splits/test.jsonl",
                            predictions=ROOT / "results" / source_predict_path,
                            processed=ROOT / "data/processed" / source_processed_path,
                            judges=ROOT / "eval/prompt_success_judgements.jsonl",
                            output=ROOT / "eval" / save_name, allow_unverified=allow_unverified)
    print(f"Saved timing-aware ASR: {ROOT / 'eval' / save_name}")
    return report


def eval_with_baseline(*, allow_unverified=False):
    paths = {"combined": "combined_predictions.jsonl", "raguard": "raguard_predictions.jsonl",
             "sentinel": "sentinel_predictions.jsonl", "toolparsing": "tool_result_parsing_predictions.jsonl"}
    results = {}
    for name, path in paths.items():
        source = ROOT / "models/baseline" / path
        if not source.exists():
            print(f"Missing baseline predictions, not scored: {source}")
            continue
        results[name] = evaluate_files(ROOT / "data/splits/test.jsonl", predictions=source,
                                       processed=ROOT / "data/processed/test_processed.jsonl",
                                       judges=ROOT / "eval/prompt_success_judgements.jsonl",
                                       allow_unverified=allow_unverified)
    write_json(ROOT / "eval/eval_asr_within_baseline.json", {k: v["per_attack_type"] for k, v in results.items()})
    write_json(ROOT / "eval/asr_baseline_details.json", results)
    return results
