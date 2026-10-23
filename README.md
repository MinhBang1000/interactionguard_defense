# InteractionGuard

Config-driven offline evaluation of a three-layer defense over stored LLM-agent
interaction traces. The runner does not intervene in a live agent loop.

## Start

```bash
conda activate agent_defense
python main.py
```

Edit **`config/experiments.yaml`** before running. This is the sole configuration
source for the new runner, including the exact saved L3 system prompt and schema.
There are no parameter questions while an experiment runs. Existing output
requires an explicit overwrite confirmation (default: No).

| Item | Operation | Required completed artifacts |
| --- | --- | --- |
| 0 | Full Dataset Preparation | Original raw traces, preserved split membership, judge cache/reviews |
| 1 | Train and Calibrate Encoder + Layer 1 | 0 |
| 2 | Layer Configuration Comparison | 0, 1; Stage 2 also requires compatible Stage 1 |
| 3 | Channel-Specific Baseline and ASR Comparison | 0, completed 2 including L1-L2-L3 |
| 4 | Same-L3 Channel vs Full-History Comparison | 0 |
| 5 | Layer 3 Backend Comparison | Stage 1 of 2; configured order must exist there |
| 6 | Layer 1 Threshold Sweep | Stage 1 of 2 |
| 7 | Benign-Ratio Sweep | Stage 1 of 2 |
| 8 | Backup Snapshot | Existing results/config; checkpoints optional |
| Q | Exit | None |

A selected experiment never silently runs prerequisites. Missing, incomplete or
incompatible artifacts produce an English error identifying what to run first.
Noninteractive item selection is also available:
`python main.py --run 2`. Parameters still come only from YAML.

## Environment and Models

Set `runtime.python`, `runtime.l3_python` and the physical GPU index in YAML.
The ML and vLLM environments are separate. The supplied paths reflect the
reference server; change paths when moving machines. Required installed packages
include PyYAML, NumPy, PyTorch, Transformers, datasets, scikit-learn,
sentence-transformers, and a compatible vLLM environment (reference: 0.11.2).
Matplotlib is optional for PNG figures; JSON/CSV reports are always produced.

Model snapshots must already be cached at the configured immutable revisions.
The runner uses local-only/offline HF loading: it does not silently download
weights. A missing model is an error, not a fallback to another checkpoint.
OpenAI-compatible L3 backends can be configured explicitly with
`backend: openai-compatible`, `base_url` and `api_key_env`; credentials come from
the environment, not YAML or saved artifacts.

The default profile retains ModernBERT-base (8192-token CLS embeddings), a
benign-trained AE, frozen `deepset/deberta-v3-base-injection` with JSON-decoded
512-token/64-overlap windows (mean within payload, max across payloads), and the
saved Llama 3.1 8B authorization-audit prompt/schema. L3 ORs its five attack flags
and requires confidence >= 0.5. Exact payload clipping remains 1800 prompt
characters, 1200 characters/block, first two blocks/channel. Context capacity
does not mean every original trace is passed in full.

## Dataset Preparation

Supply the original files in `data/raw/`:

```text
benign_traces.jsonl
prompt_injection_traces.jsonl
rag_poison_traces.jsonl
tool_injection_traces.jsonl
correlated_injection_traces.jsonl
```

Keep the existing `data/splits/split_manifest.json` or recoverable split files.
Item 0 preserves episode membership/order; it does not silently resplit.
Its flow is the previous menu 10 -> 3:
raw -> ASR annotation/judge -> splits -> normalization/prefix expansion ->
fixed encoder/AE datasets -> validation/reporting.

Original raw, split and judge inputs stay unchanged. New data and copied judge
cache/reviews are published under `results/research_runner/dataset/`.
Missing prompt-success judgments may invoke the configured Llama judge during a
real Item 0 run. `data.judge.skip: true` disables those calls; missing evidence
then remains UNKNOWN. The success judge is distinct from the L3 defense.

The default reference test SHA-256 is
`699acf0f601c1ed458d68a917aee175ce8a2688e8fc4486584bf34daf0ff3ed9`.
A mismatch stops preparation. Current source data have 11,342/4,131/5,445
train/validation/test prefixes. Encoder exports contain 9,843/3,514 rows;
AE exports contain 8,114/2,813 benign rows. Correlated and unresolved training
annotations are excluded. Preserve the overlap disclosure: test content is not
claimed entirely unseen.

## Training and Decisions

Item 1 trains seeds 42-46 once, preserving checkpoints for later experiments.
Encoder training uses both prefix labels; AE training uses label 0 only.
Correlated is held out from project adaptation/calibration. This is not a
leave-one-family-out evaluation or proof of upstream pretraining independence.

AE model selection follows the reference validation recipe (recall floor 0.98).
The deployment L1 gate is recalibrated separately for each seed at recall >=
0.999 using the recorded validation-F1 constrained objective. L2 has one
validation-calibrated threshold under empirical FPR <= 0.01, shared across seeds.
Threshold ties are not split. Test labels are not used for selection.

| Layer | Nonterminal behavior | Terminal behavior |
| --- | --- | --- |
| L1 | error > threshold: Check; otherwise Forward | error > threshold: Stop; otherwise Forward |
| L2 | probability > threshold: Stop; otherwise Check | probability > threshold: Stop; otherwise Forward |
| L3 | Only evaluated as terminal auditor | Saved flag/confidence decision |

Thus L1-L2 blocks their positive intersection; L2-L1 blocks their positive union.
Full L1-L2-L3 keeps L1 as screening only. There is no global terminal
`Check -> Forward` rule.

## Two-Stage Configuration Comparison

Set `experiments.configuration.stage` to `stage1`, `stage2`, or `full`.
Stage 1 saves all L1/L2 scores, per-seed/per-order routing and exact L3 requests.
Its subprocess exits before Stage 2 starts; CUDA contexts are not retained in
the menu process. GPU preflight waits only for the configured interval and
never terminates another user's jobs. A busy GPU stops the run.

Stage 2 verifies Stage 1 artifacts, performs L3 inference and joins by IDs.
Standalone L3 receives the whole test set. Backend comparison reuses the same
pending sets in Stage 1 for every backend within a seed.

A completed Stage 1 can be reused independently. Set the relevant experiment's
`resume: true` to reuse verified completed output; configuration Stage 2 is
still separate. An interrupted, unpublished stage is rerun rather than counted
as complete. No partial/mismatched result is silently scored benign.

## Experiments and Reporting

Item 3 uses Sentinel (prompt), RAGuard (retrieval), Prompt Guard 2 (tool payload),
their OR combination, no defense and IG. Prompt Guard 2 replaces ToolResultParsing;
this is not a reproduction of ToolResultParsing. Baselines are offline channel
adaptations. Individual payload/window scores and channel verdicts are retained.

Item 4 compares the same L3 on channels, channel OR and history, with explicit
original-prompt anchoring. It saves complete-fragment correlated candidates;
both fragments existing does not establish their joint causal necessity.

Item 5 changes only L3 backend over fixed upstream routing. The supplied local
candidates must be available in their configured environments. Item 6 sweeps
validation-derived L1 gates, not test-selected numerical thresholds. Item 7
samples 90/92/94/96/98% benign prefixes without replacement, with reproducible
IDs and attack-category stratification, keeping thresholds fixed per seed.
Because these are partial-prefix workloads, whole-episode ASR/false-blocking is
reported in Item 3, not inferred from incomplete Item 7 episodes.

Every result directory contains effective config, completion/file-hash manifest,
worker logs and experiment-specific request/score/decision records. Reports
include prefix metrics, routing traffic, per-category results and episode ASR
where full episodes are covered. ASR uses `eval/asr.py`: prevention requires a
block before the safe boundary. Natural failures, pre-injection blocks, late
detections, benign episode false-blocks, macro and pooled ASR remain distinct.

Summary JSON/CSV contains means and sample standard deviations. Figures have no
chart title (use LaTeX captions); sweep figures combine F1/traffic on a percentage
axis and component-based pipeline time on a separately scaled right axis.

Five training seeds are not five retrainings of a frozen classifier. Serving,
sampling and training variation must be identified separately. Accuracy
deduplication, when enabled by the timing configuration, is not an independent
inference repeat. The default uncached timing path does not reuse verdicts.
Staged timing is a **component-based pipeline estimate**, not direct end-to-end
latency or production throughput; model-loading time is excluded.

## Overwrite and Backup

Results publish transactionally from staging. A failed replacement leaves the
previous complete output intact. A config/input/code mismatch invalidates reuse;
never manually edit a completed directory.

Item 8 creates a timestamped ZIP in `backup/` using the configured scope. It
includes run-effective configs, checksums and a backup manifest, validates ZIP
integrity/content, and excludes secrets, environment/cache directories and
nested ZIPs. Set `backup.include_checkpoints: true` for model recovery; otherwise
weights are excluded. Results without checkpoints do not recreate trained
weights. Original data are included only if explicitly selected through safe
`backup.include_paths`.

## CPU Functional Tests

```bash
python -m unittest discover -s research_runner/tests -v
```

Tests use temporary synthetic datasets and fake model methods. They exercise
real preparation, calibration, routing, worker dispatch, artifact publication,
ASR and backup. No training, downloads, API requests or GPU inference are run.
Optional read-only reference checks are skipped when local historical artifacts
are absent. Passing these tests does not establish real-model accuracy or
successful full experiments; the user performs those runs.

## Scope and Legacy Code

`main.py` exposes only the new runner. Old detector/training/data modules and
standalone experiment scripts remain where reused or independently callable;
their `config/config.py`, `detection_config.yaml` and old menus are not the new
runner's parameter source. Do not use old default profiles to reproduce this
campaign. Historical results, paper PDFs, model checkpoints, and the independent
`agentdojo/` pipeline are not deleted. `docs/` remains local.

The evaluation is exploratory after earlier test inspection; synthetic few-shot
examples do not establish untouched-test independence. Deepset's upstream data
overlap must be disclosed. Auditor-targeted adaptive attacks, online defended
agent execution and arbitrary-length-history robustness are not evaluated.
