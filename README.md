# InteractionGuard

## Functional Names (Naming Version 2)

The research runner uses stable component identities, independent of order:

| Runtime name | Mechanism | Legacy artifact name | Paper label |
| --- | --- | --- | --- |
| Screen | Frozen Deepset known-pattern classifier | L2 | L1 |
| Gate | ModernBERT + AutoEncoder anomaly routing | L1 | L2 |
| Audit | LLM semantic auditor | L3 | L3 |

For example, legacy `L2-L1-L3` is `Screen-Gate-Audit`. Only labels change:
models, scores, thresholds, routing and verdicts retain their original meaning.
The fixed mapping is snapshotted from `config/experiments.yaml` (`naming`).
`research_runner.naming.paper_order` translates labels for paper exports.

Runner-owned preprocessing, ASR, training and detector implementations live in
`research_runner/implementation/`. They do not import the old project modules.
Historical side tests and legacy sources keep their original names and contents.
The existing Audit Python environment path is an environment reference, not a
component label; its legacy directory name is intentionally unchanged.

The one-way migration utility is `tools/migrate_runner_naming.py`. Use
`python tools/migrate_runner_naming.py --help` for its dry-run/apply interface.
It requires a verified pre-migration inventory/backup (`before.json`); it will
not infer a missing backup. Real migration provenance is stored in
`results/research_runner/.naming_migration.json`, with a second copy in the backup.
Do not run experiments or edit runner sources while migrating. On the current
installation the migration is already complete; do not rerun completed experiments
just because their component names changed. To inspect/check an existing migration:

```bash
python tools/migrate_runner_naming.py artifacts --backup backup/naming_v2_20261024T204153
```

On an unmigrated installation this is a dry-run; add `--apply` only after checking
the backup and plan. On an already migrated installation it validates compatibility
and returns `already_version_2`, without applying names twice.

Original source hashes and frozen policy identities remain historical evidence,
not claims that models were retrained with renamed source files. Frozen Audit
input strings, including the historical clipping marker, remain byte-identical.

Artifacts are staged and checked before publication. Publication renames the old
result root to a `.naming-v1-recovery-*` sibling first, then installs the verified
new root. On a publication exception it restores the old root. After interruption,
retain both directories and the backup; never run experiments against a missing
or partial root. Recovery is by restoring the complete old directory (and its
matching code/config snapshot), not merging files across naming versions.

Config-driven offline evaluation of a three-layer defense over stored LLM-agent
interaction traces. The runner does not intervene in a live agent loop.

## Start

```bash
conda activate agent_defense
python main.py
```

Edit **`config/experiments.yaml`** before running. This is the sole configuration
source for the new runner, including the exact saved Audit system prompt and schema.
There are no parameter questions while an experiment runs. Existing output
requires an explicit overwrite confirmation (default: No).

| Item | Operation | Required completed artifacts |
| --- | --- | --- |
| 0 | Full Dataset Preparation | Original raw traces, preserved split membership, judge cache/reviews |
| 1 | Train and Calibrate Encoder + Gate | 0 |
| 2 | Layer Configuration Comparison | 0, 1; Stage 2 also requires compatible Stage 1 |
| 3 | Channel-Specific Baseline and ASR Comparison | 0, completed 2 including every requested order/seed |
| 4 | Same-Audit Channel vs Full-History Comparison | 0 |
| 5 | Audit Backend Comparison | Stage 1 of 2 including Screen-Gate-Audit for every seed |
| 6 | Gate Threshold Sweep | Stage 1 of 2 |
| 7 | Benign-Ratio Sweep | Stage 1 of 2 |
| 8 | Backup Snapshot | Existing results/config; checkpoints optional |
| Q | Exit | None |

A selected experiment never silently runs prerequisites. Missing, incomplete or
incompatible artifacts produce an English error identifying what to run first.
Noninteractive item selection is also available:
`python main.py --run 2`. Parameters still come only from YAML.

## Environment and Models

Set `runtime.python`, `runtime.audit_python` and the physical GPU index in YAML.
The ML and vLLM environments are separate. The supplied paths reflect the
reference server; change paths when moving machines. Required installed packages
include PyYAML, NumPy, PyTorch, Transformers, datasets, scikit-learn,
sentence-transformers, and a compatible vLLM environment (reference: 0.11.2).
Matplotlib is optional for PNG figures; JSON/CSV reports are always produced.

Model snapshots must already be cached at the configured immutable revisions.
The runner uses local-only/offline HF loading: it does not silently download
weights. A missing model is an error, not a fallback to another checkpoint.
OpenAI-compatible Audit backends can be configured explicitly with
`backend: openai-compatible`, `base_url` and `api_key_env`; credentials come from
the environment, not YAML or saved artifacts.

The default profile retains ModernBERT-base (8192-token CLS embeddings), a
benign-trained AE, frozen `deepset/deberta-v3-base-injection` with JSON-decoded
512-token/64-overlap windows (mean within payload, max across payloads), and the
saved Llama 3.1 8B authorization-audit prompt/schema. Audit ORs its five attack flags
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
Final provenance paths are assigned before dataset checksums are collected;
publication does not re-serialize the hashed metadata. Both dataset reports
describe the published files' SHA-256, byte sizes and JSONL row counts, while
the outer artifact manifest additionally covers the reports/effective config.
Missing prompt-success judgments may invoke the configured Llama judge during a
real Item 0 run. `data.judge.skip: true` disables those calls; missing evidence
then remains UNKNOWN. The success judge is distinct from the Audit defense.

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
The deployment Gate gate is recalibrated separately for each seed at recall >=
0.999 using the recorded validation-F1 constrained objective. Screen has one
validation-calibrated threshold under empirical FPR <= 0.01, shared across seeds.
Threshold ties are not split. Test labels are not used for selection.

| Layer | Nonterminal behavior | Terminal behavior |
| --- | --- | --- |
| Gate | error > threshold: Check; otherwise Forward | error > threshold: Stop; otherwise Forward |
| Screen | probability > threshold: Stop; otherwise Check | probability > threshold: Stop; otherwise Forward |
| Audit | Only evaluated as terminal auditor | Saved flag/confidence decision |

Thus Gate-Screen blocks their positive intersection; Screen-Gate blocks their positive union.
Full Gate-Screen-Audit keeps Gate as screening only. There is no global terminal
`Check -> Forward` rule.

## Two-Stage Configuration Comparison

Set `experiments.configuration.stage` to `stage1`, `stage2`, or `full`.
Stage 1 saves all Gate/Screen scores, per-seed/per-order routing and exact Audit requests.
Its subprocess exits before Stage 2 starts; CUDA contexts are not retained in
the menu process. GPU preflight waits only for the configured interval and
never terminates another user's jobs. A busy GPU stops the run.

Stage 2 verifies Stage 1 artifacts, performs Audit inference and joins by IDs.
Standalone Audit receives the whole test set. Backend comparison reuses the same
pending sets in Stage 1 for every backend within a seed.

A completed Stage 1 can be reused independently. Set the relevant experiment's
`resume: true` to reuse verified completed output; configuration Stage 2 is
still separate. An interrupted, unpublished stage is rerun rather than counted
as complete. No partial/mismatched result is silently scored benign.

## Experiments and Reporting

The primary configuration is explicitly `primary_order: Screen-Gate-Audit`.
Item 2 and Item 7 retain their existing scientific settings. Edit the following
sections of the **same** YAML for the new comparisons:

```yaml
primary_order: Screen-Gate-Audit
experiments:
  baselines:                 # Item 3
    orders: [Screen-Gate-Audit, Gate-Screen-Audit]
  backend:                   # Item 5: one fixed upstream order
    order: Screen-Gate-Audit
    candidates:              # retain each candidate's actual model/revision/settings
      - name: llama31_8b
        backend: vllm
        model: meta-llama/Llama-3.1-8B-Instruct
  threshold:                 # Item 6
    orders: [Screen-Gate-Audit, Gate-Screen-Audit]
```

This is a structural excerpt, not a replacement for the complete pinned YAML.
Item 3/6 require nonempty, unique, valid orders including the primary order.
Item 5 rejects other orders and duplicate/unsafe candidate names. Item 4 remains
a channel/history ablation, not a cascade-order comparison. No new menu entries
or runtime parameter questions are introduced.

Item 3 uses Sentinel (prompt), RAGuard (retrieval), Prompt Guard 2 (tool payload),
their OR combination, no defense and IG. Prompt Guard 2 replaces ToolResultParsing;
this is not a reproduction of ToolResultParsing. Baselines are offline channel
adaptations. Individual payload/window scores and channel verdicts are retained.
The three baseline detectors run once, not once per IG order or seed. Their
reports, combined OR and no defense remain at `baselines/<detector>/`.
IG decisions are copied from compatible Item 2 results without new model calls:
`baselines/seed_<seed>/InteractionGuard/<order>/`. Each `reuse.json` records the
source checksums and explicitly labels inherited latency estimates. Summary
conditions are `InteractionGuard/<order>`; deterministic baselines have one
accuracy trial rather than invented five-run variability.

Item 4 compares the same Audit on channels, channel OR and history, with explicit
original-prompt anchoring. It saves complete-fragment correlated candidates;
both fragments existing does not establish their joint causal necessity.

Item 5 changes only Audit backend over fixed **Screen-Gate-Audit** Stage 1 routing.
It preflights every seed before inference. Candidates receive identical **ordered**
prefix IDs and payloads per seed, never regenerated Screen/Gate scores. Outputs
remain `backend/seed_<seed>/<candidate>/`, with pending-set/routing hashes and
ordered IDs in `routing_identity.json`. Reports score the entire pipeline, not
only Audit's pending subset. Local candidates must already be cached; an explicitly
configured OpenAI-compatible candidate uses the same pending set and requires its
own API environment variable. No candidate is silently substituted.

Item 6 reuses Stage 1 Gate test/validation scores and the shared Screen scores.
Each seed's thresholds are selected **only on validation**, using the existing
recall-floor/F1 policy. Both orders use the same threshold and ordered full-test
IDs at each operating point. Screen/Audit models, Screen threshold and Audit
policy stay fixed; new routing determines which Audit calls are needed. Outputs:
`threshold/seed_<seed>/<operating_point>/<order>/`. Shared and per-order
`operating_point.json` record score hashes, calibration and paired-workload hashes.
Audit request/response records identify seed, operating point and order.
Summary JSON/CSV includes F1, precision/recall/FPR/FNR, all component processing
ratios, estimated pipeline time, category/macro/pooled ASR and benign episode
false-blocking; detailed ASR files retain counts and detection timing.
The main figure compares orders; `<order>/comparison.png` retains the complete
F1/traffic/latency chart for each order. Legends sit above the data; no titles.

Item 7
samples 90/92/94/96/98% benign prefixes without replacement, with reproducible
IDs and attack-category stratification, keeping thresholds fixed per seed.
Item 7 now compares `experiments.benign.orders: [Gate-Screen-Audit, Screen-Gate-Audit]` on the
exact same ordered prefix IDs for each seed/ratio. Sampling happens once before
the order loop; thresholds are not retuned. It reuses Stage 1 Gate/Screen scores and
runs Audit only for each condition's pending set under the configured timing policy.
Results are separated as `benign/seed_<seed>/benign_<ratio>/<order>/`; shared
`workload.json` records and paired-workload hashes prove the pairing. Each order
also records benign prefixes forwarded by Gate at the front gate. Summaries group
by `benign_<ratio>/<order>`. The main figure compares both orders' F1/estimated
latency; each order retains a separate F1/traffic/latency figure, without titles.
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

## Live Progress

Every menu item (0-8) prints live `[IG]` updates without extra parameters:
current stage, seed/condition, completed/total units, a percentage bar, elapsed
time and estimated remaining time when the total is known. Training reports
encoder optimizer steps/validation batches and AE candidates/epochs/batches;
evaluation reports Gate prefixes, Screen windows, Audit requests/batches and RAG documents.
Dataset preparation reports annotation, split, prefix expansion and exports.
Backup reports copied bytes/files and ZIP verification.

Loading, preprocessing internals, or long batches with no known completion count
show a heartbeat after about five seconds instead of an invented percentage.
Heartbeat means the runner is alive, not that a backend is guaranteed healthy.
Percentages and ETA refer to the **current phase**, not the whole campaign;
early stopping can finish training below its planned step/epoch budget.

Worker progress is streamed from `worker.log` (nested training uses
`model/training.log`). Logs remain available; scientific artifacts are not
rewritten to track live status. The latest menu status and append-only event
history are stored separately at:

```text
results/research_runner/.progress/<experiment>/progress.json
results/research_runner/.progress/<experiment>/progress.jsonl
```

For example, monitor item 2 from another terminal:

```bash
tail -f results/research_runner/.progress/configuration/progress.jsonl
```

Workers also save local `progress/` records inside their staged/published output.
Live `.progress/` files are excluded from backups; stable worker logs and status
records remain subject to the configured backup scope. Progress does not change
thresholds, inputs, batching, random seeds, ASR rules or GPU stage ordering.

## Overwrite and Backup

Results publish transactionally from staging. A failed replacement leaves the
previous complete output intact. A config/input/code mismatch invalidates reuse;
never manually edit a completed directory.

An explicitly audited code-only upgrade may record a compatibility receipt in
`results/research_runner/.compatibility/`, without rewriting completed artifacts.
Receipts bind the original manifest SHA/signature to the exact upgraded signature
(including current code, scientific settings and inputs). They do not disable file
integrity checks or accept later code/model/data changes. The benign-order upgrade
preserves the verified Item 0-2 prerequisites this way; it does not retrain them.
Keep these receipts in result backups; do not fabricate or manually update them.

The multi-configuration upgrade uses `tools/upgrade_multiconfig.py` with an
explicit **pre-edit** capture. On this installation the verified capture is
`backup/multiconfig_20261024_d0c5244/`. It contains original source/config and
manifest/file hashes, plus `upgrade_record.json` after application. Completed
dataset/training/configuration/benign files are not rewritten; only exact-bound
compatibility receipts are published. The original naming-migration backup still
contains the full result/weight backup. No rerun of Items 0/1/2/7 is required for
this audited upgrade. A model/data/threshold/prompt change still invalidates reuse.

For maintainers only: capture BEFORE editing with
`python tools/upgrade_multiconfig.py capture --backup <new-directory>`; inspect
`python tools/upgrade_multiconfig.py apply --backup <directory>` (dry-run), then
add `--apply` only for the reviewed code-only upgrade. The tool rejects changes
outside the declared source/config scope, verifies original files, and restores
previous receipts on verification failure. Do not recapture or reseal to bypass
a scientific incompatibility, or edit sources during sealing. Normal experiment
execution never runs this upgrade automatically.

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
