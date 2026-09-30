# Agent Defense: Multi-Layer Security for LLM Agents

This repository implements a research-oriented defense framework for LLM-based agents against:

- prompt injection
- tool injection
- RAG poisoning
- correlated multi-channel attacks

The system combines three layers:

- Layer 1: AutoEncoder-based anomaly screening over encoder embeddings
- Layer 2: fine-tuned classifier for known malicious traces
- Layer 3: LLM-based semantic security auditor

It also includes:

- data split and preprocessing pipelines
- model training scripts
- baseline defenses for Issue 3
- evaluation scripts
- figure generation for the paper/report

## Repository Layout

```text
agent_defense/
├── config/                  # YAML and Python config
├── data/
│   ├── raw/                 # Raw trace JSONL files
│   ├── splits/              # Train/val/test trace splits
│   └── processed/           # Prefix-expanded processed samples
├── eval/                    # Evaluation outputs and scripts
├── models/
│   ├── detectors/           # Layer 1 / 2 / 3 + pipeline
│   ├── baseline/            # RAGuard / Sentinel / Tool Result Parsing baselines
│   └── layer1/              # Saved encoder / autoencoder artifacts
├── results/                 # Detection outputs, sweep outputs, figures
├── training/                # Training scripts
├── utils/                   # Data, metrics, plotting, menu helpers
└── main.py                  # Main interactive entry point
```

## Data Flow

The codebase follows this high-level workflow:

1. Raw traces are stored in `data/raw/*.jsonl`.
2. `data/data_split.py` creates train/val/test splits in `data/splits/`.
3. `data/data_preprocess.py` converts each trace into one or more sequence samples in `data/processed/`.
4. Training scripts build Layer 1 and Layer 2 artifacts.
5. `main.py` runs evaluation, baseline tests, ASR analysis, and figure generation.

Processed samples use a sequence format such as:

```text
[PROMPT]
...

[SEP]

[RETRIEVE]
...

[SEP]

[TOOL: get_information]
...

[SEP]

[AGENT]
...

[EOS]
```

## Defense Modes

The detection pipeline supports six modes from `config/test_modes.yaml`:

- `1`: Layer 1 only
- `2`: Layer 2 only
- `3`: Layer 3 only
- `10`: Layer 1 -> Layer 2
- `11`: Layer 1 -> Layer 2 -> Layer 3
- `12`: Layer 2 -> Layer 1 -> Layer 3

Important:

- `10` is a 2-layer pipeline
- `11` is the full 3-layer pipeline
- figures that refer to the "3-layer" defense need results from mode `11`

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Recommended:

- Python 3.10+
- PyTorch
- Transformers
- scikit-learn
- tqdm
- OpenAI-compatible client dependencies

## Environment Variables

Create a `.env` file in the project root.

Typical variables used by this repository:

```env
OPENAI_API_KEY=...
OPENROUTER_API_KEY=...
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1

TOOL_RESULT_EXTRACT_METHOD_LIST=ParseData,CheckTool
TOOL_RESULT_EXTRACT_PARSE_DATA_FULL_CONVERSATION=False
TOOL_RESULT_EXTRACT_CHECK_TOOL_FILTER=True
```

Notes:

- Layer 3 and some evaluation scripts require an LLM API key.
- The Tool Result Parsing baseline now follows the study-style OpenRouter workflow.

## Run the Project

Launch the interactive menu:

```bash
python main.py
```

## Main Menu Reference

When `main.py` starts, you will see this main menu.

### `1` Run multi-layer evaluation

What it does:

- runs the detection pipeline on `data/processed/test_processed.jsonl`
- asks you to choose a defense mode: `1`, `2`, `3`, `10`, `11`, or `12`
- saves the output to `results/mode_<id>_results.json`

Use this when:

- you want raw detection results for one pipeline mode
- you need `mode_11_results.json` for full 3-layer ASR evaluation

### `2` Evaluate prompt attacks (no defense, LLM judge)

What it does:

- evaluates prompt injection success without defense using an LLM judge
- writes prompt failure cases used later by the no-defense evaluator

Use this when:

- you want prompt-attack success labels refreshed from scratch

### `3` Evaluate all attacks (no defense, manual)

What it does:

- generates `eval/eval_asr_without_defense.json`
- computes no-defense ASR for `prompt`, `rag`, `tool`, and `correlated`

Dependency note:

- for prompt attacks, this uses the failure cases from option `2`

Use this when:

- you need the "No Defense" line/bar data for Issue 3 figures

### `4` Evaluate attacks (with 3-layer and baseline defenses)

What it does:

- evaluates the 3-layer defense and baseline defenses
- writes/update:
  - `eval/eval_asr_within_defense.json`
  - `eval/eval_asr_within_baseline.json`

Important:

- the 3-layer part expects `results/mode_11_results.json`
- the baseline part expects baseline prediction files to already exist

Use this when:

- you want ASR after defense
- you want to refresh the data used by Issue 3 figures

### `5` Generate evaluation figures

What it does:

- generates saved figures in `results/`
- includes Issue 1, Issue 2, and Issue 3 figure outputs

Important:

- figures only reflect whatever evaluation files already exist
- this option does not re-run detectors automatically

### `6` Fine-tune Layer 1 encoder (ModernBERT)

What it does:

- fine-tunes the Layer 1 encoder backbone
- saves artifacts into the Layer 1 model directory

Use this when:

- you want to retrain the embedding encoder used by Layer 1

### `7` Train Layer 1 AutoEncoder

What it does:

- trains the Layer 1 autoencoder on processed benign data

Use this when:

- you already have the encoder and want to rebuild the anomaly detector

### `8` Fine-tune Layer 2 model (ProtectAI)

What it does:

- fine-tunes the Layer 2 classifier
- saves the Layer 2 model directory used by the pipeline

Use this when:

- you want to retrain the supervised malicious detector

### `9` Find optimal Layer 1 threshold

What it does:

- searches for a Layer 1 threshold with zero false negatives while minimizing false positives

Use this when:

- you want to tune the Layer 1 anomaly threshold

### `10` Run baseline tests (Issue 3)

What it does:

- opens a baseline submenu
- runs baseline detectors and writes baseline prediction files

This does not directly generate Issue 3 charts. It creates the prediction artifacts that later evaluation depends on.

### `11` Evaluate baseline results (Issue 3)

What it does:

- evaluates baseline prediction files
- writes ASR reports used by downstream figure generation

Use this when:

- you changed a baseline prediction file and want figures/eval to reflect the new numbers

### `12` Sweep L1 threshold -> analyze L3 load (Issue 2)

What it does:

- runs threshold sweeps for mode `11` or `12`
- stores outputs under `results/threshold_sweep/`

Use this when:

- you want Issue 2 threshold sensitivity data

### `13` Sweep data distribution -> analyze L3 load (Issue 2)

What it does:

- sweeps benign/attack mix ratios for mode `11` or `12`
- stores outputs under `results/distribution_sweep/`

Use this when:

- you want Issue 2 distribution sensitivity data

### `14` High-benign distribution sweep -> compare mode 11 vs 12 (Issue 2)

What it does:

- runs a separate high-benign sweep without replacing the legacy distribution sweep
- evaluates both mode `11` and mode `12` on the exact same sampled subsets for fair comparison
- reads the Layer 1 threshold directly from `config/detection_config.yaml`
- reads the fixed sample size and default benign range from `config/config.py`
- stores outputs under `results/distribution_sweep_high_benign/`

Sampling policy:

- fixed sample size is set to `2772`
- default benign range is `90%` to `99%`
- malicious samples are selected as evenly as possible across:
  - `prompt`
  - `rag`
  - `tool`
  - `correlated`
- if a perfect balance does not fill the required malicious budget, the remainder is sampled randomly from the remaining malicious pool

Use this when:

- you want a more realistic high-benign workload for Issue 2
- you want a direct mode `11` vs `12` comparison under the same sampled inputs

## Baseline Submenu Reference

If you choose main menu option `10`, you enter the baseline submenu.

### `10 -> 1` RAGuard

Runs the RAG-only baseline and updates:

- `models/baseline/raguard_predictions.jsonl`
- `models/baseline/raguard_metrics.json`
- `models/baseline/raguard_thresholds.json`

### `10 -> 2` Tool Result Parsing

Runs the Tool Result Parsing baseline and updates:

- `models/baseline/tool_result_parsing_predictions.jsonl`
- `models/baseline/tool_result_parsing_metrics.json`
- `models/baseline/tool_result_parsing_thresholds.json`

This baseline now follows the study-style `ParseData -> CheckTool` workflow while keeping the legacy output file contract expected by `main.py`.

### `10 -> 3` Prompt Sentinel

Runs the prompt-channel baseline and updates:

- `models/baseline/sentinel_predictions.jsonl`
- `models/baseline/sentinel_metrics.json`
- `models/baseline/sentinel_thresholds.json`

### `10 -> 4` Combined Defense

What it does:

- checks whether the three baseline prediction files already exist
- automatically runs missing detectors if needed
- merges prompt, RAG, and tool baseline outputs into:
  - `models/baseline/combined_predictions.jsonl`

Use this when:

- you want the `Combined` baseline for Issue 3

## Common Workflows

This section explains which menu actions to take for common tasks.

### Build the full 3-layer result from scratch

Menu path:

- `1 -> choose 11`

Result:

- creates `results/mode_11_results.json`

### Refresh only baseline predictions

Menu path:

- `10 -> 1`
- `10 -> 2`
- `10 -> 3`

If you also need the merged baseline:

- `10 -> 4`

### Recompute baseline evaluation after changing a baseline

Menu path:

- `11`

Use this when:

- `tool_result_parsing_predictions.jsonl` changed
- `raguard_predictions.jsonl` changed
- `sentinel_predictions.jsonl` changed
- `combined_predictions.jsonl` changed

### Generate the ASR line chart quickly from existing baseline outputs

Menu path:

- `10 -> 4`
- `4`
- `5`

This is the shortest path only if all required prerequisite files already exist, especially:

- `results/mode_11_results.json`
- `eval/eval_asr_without_defense.json`

### Generate the ASR line chart from a clean or uncertain state

Menu path:

- `2`
- `3`
- `1 -> choose 11`
- `10 -> 4`
- `4`
- `5`

Why this order:

- `2` refreshes prompt no-defense judging
- `3` writes `eval_asr_without_defense.json`
- `1 -> 11` writes the full 3-layer result
- `10 -> 4` creates the combined baseline prediction
- `4` writes defense and baseline ASR evaluation files
- `5` generates the Issue 3 figure

### Update Issue 3 after rerunning only Tool Result Parsing

Menu path:

- `10 -> 2`
- `10 -> 4`
- `11`
- `5`

Why:

- `10 -> 2` updates tool baseline predictions
- `10 -> 4` refreshes `combined_predictions.jsonl`
- `11` regenerates baseline ASR reports
- `5` redraws figures

### Evaluate baseline + 3-layer ASR after rerunning mode 11

Menu path:

- `1 -> choose 11`
- `4`
- `5`

### Generate Issue 2 threshold/load figures

Menu path:

- `12 -> choose 11 or 12`
- `5`

### Generate Issue 2 distribution sweep figures

Menu path:

- `13 -> choose 11 or 12`
- `5`

### Generate Issue 2 high-benign comparison figures

Menu path:

- run the high-benign sweep first
- then generate figures normally

This adds new figures such as:

- `results/issue2_high_benign_l3_ratio_compare.png`
- `results/issue2_high_benign_l3_latency_compare.png`
- `results/mode_11_high_benign_distribution_sensitivity.png`
- `results/mode_12_high_benign_distribution_sensitivity.png`

## Key Output Files

### Detection pipeline outputs

- `results/mode_10_results.json`
- `results/mode_11_results.json`
- `results/mode_12_results.json`

### Baseline outputs

- `models/baseline/raguard_predictions.jsonl`
- `models/baseline/tool_result_parsing_predictions.jsonl`
- `models/baseline/sentinel_predictions.jsonl`
- `models/baseline/combined_predictions.jsonl`

### Evaluation outputs

- `eval/eval_asr_without_defense.json`
- `eval/eval_asr_within_defense.json`
- `eval/eval_asr_within_baseline.json`

### Figure outputs

Examples:

- `results/issue3_ASRline_chart.png`
- threshold sweep figures
- distribution sweep figures
- high-benign distribution sweep figures

## Important Notes

- `issue3_ASRline` does not read `models/baseline/tool_result_parsing_metrics.json` directly.
- It reads `eval/eval_asr_within_baseline.json`.
- Therefore, after changing a baseline prediction file, you must re-run baseline evaluation before regenerating figures.
- The full 3-layer ASR result depends on mode `11`, not mode `10`.
- The high-benign sweep uses a fixed total size of `2772`, which is the maximum size that still fits the current test split at `99%` benign under the repository's `int(total_size * ratio)` sampling rule.

## Quick Reminder

If you want a specific function, use these menu paths:

- Full 3-layer detection result: `1 -> 11`
- Run Tool Result Parsing baseline: `10 -> 2`
- Rebuild Combined baseline: `10 -> 4`
- Refresh baseline ASR reports: `11`
- Refresh defense + baseline ASR reports: `4`
- Generate ASR line chart from ready data: `10 -> 4 -> 4 -> 5`
- Generate ASR line chart from scratch: `2 -> 3 -> 1 -> 11 -> 10 -> 4 -> 4 -> 5`

## Entry Points

Primary entry:

```bash
python main.py
```

Data processing:

```bash
python -m data.data_split
python -m data.data_preprocess
```

## Project Status

This is a research codebase rather than a production package. Some scripts are experimental or paper-oriented, and several workflows depend on existing intermediate files. The menu reference above is the safest way to operate the repository consistently.

## Repository History / Migration Note

2026-09-30: active development moved to a new, cleaned-up repository:
`https://github.com/MinhBang1000/interactionguard_defense`. This repository
(`agent_defense`, previously pushed to `https://github.com/MinhBang1000/agent_eval.git`)
is frozen at its last commit and kept only as a historical archive of the full
development process (all `AE_000*` branches, experimental Layer 3 variants,
Dojo integration work, etc.). Its `origin` remote was intentionally removed
(`git remote remove origin`) to prevent accidental pushes here.

To recover the link to the old GitHub remote if ever needed:

```bash
git remote add origin https://github.com/MinhBang1000/agent_eval.git
```
