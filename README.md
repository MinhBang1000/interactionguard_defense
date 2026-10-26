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
| 8 | Auditor-Targeted Case Stress Test | Completed 0, 1 and both stages of 2 for the manually selected seed/case |
| 9 | Backup Snapshot | Existing results/config; checkpoints optional |
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
    selection_policy: max_threshold_under_recall_floor
    recall_floors: [0.90, 0.95, 0.98, 0.999]
    reuse_configuration_recall_floors: [0.999]
    include_validation_max_f1: true
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
only Audit's pending subset. Newly inferred local candidates must already be cached; an explicitly
configured OpenAI-compatible candidate uses the same pending set and requires its
own API environment variable. No candidate is silently substituted.

### Item 5: Four Local Backends (2026 Profile)

The checked-in `experiments.backend.candidates` now contains exactly:

| Candidate | Checkpoint | Precision / Runtime |
| --- | --- | --- |
| `llama31_8b` | Meta Llama 3.1 8B Instruct | Exact Item 2 BF16 reference, reused without inference |
| `qwen35_9b` | Qwen/Qwen3.5-9B | BF16, text-only, thinking explicitly disabled |
| `gemma4_12b_qat_w4a16` | google/gemma-4-12B-it-qat-w4a16-ct | Official QAT W4A16 compressed-tensors, BF16 activations, text-only |
| `nemotron3_nano_4b` | nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16 | BF16, thinking explicitly disabled |

All checkpoints have immutable revisions in YAML. The three new models use a
separate `.venv/audit_2026` with vLLM **0.32.0**; the reference environment is
never upgraded. Candidate `python` overrides only that candidate's worker.
New models use the modern structured-output API and native chat templates;
the policy text/schema, payload builder, 8192-token context budget, 512-token
output cap, temperature 0 and confidence threshold 0.5 remain unchanged.
`generation_config: vllm` prevents new checkpoints from silently supplying
different sampling defaults. The historical Llama loading path stays unchanged.

From the repository root, in the existing CPU/ML environment:

```bash
# Explicit installation + pinned model downloads, but NO inference or training:
python -m research_runner.backend_setup --prepare
# Optional CPU-only local environment/snapshot validation:
python -m research_runner.backend_setup --check
# Then select menu item 5:
python main.py
```

If the models/environment are already installed, repair just the build tools:

```bash
python -m research_runner.backend_setup --repair-runtime
python -m research_runner.backend_setup --check
# Optional GPU 0 diagnostic, one synthetic request per new model, never Llama:
python -m research_runner.backend_setup --smoke
```

Modern workers prepend their own `bin` to PATH and use a venv-owned CUDA 13.0
compiler/linker SDK view (`ig_cuda_toolkit`). Pinned Ninja and CUDA build packages
are read from each candidate's `runtime_packages` in YAML. This avoids accidentally
using the host CUDA 11.5 compiler. Neither `/usr`, global shell settings, NVIDIA
drivers nor the reference Llama environment are modified. CPU preflight checks
Ninja/C++/nvcc executables, SDK headers/linker layout and CUDA version agreement
with the PyTorch build, in the same environment used by inference workers.
Modern engine initialization failures stop immediately, not after 35 repeated
model loads; failed batches never count as successful completed requests.

Nemotron alone sets `structured_outputs_config: {backend: xgrammar,
disable_any_whitespace: true}` in its candidate configuration. This prevents
unbounded formatting-whitespace generation between JSON fields, not spaces
inside string values. Prompt, content schema, 512-token output budget and verdict
rule remain unchanged. Llama/Qwen/Gemma decoding is unchanged. Disclose this
backend-specific decoding setting in Item 5; it can affect verdicts and timing,
so do not describe all candidates as having identical decoding configurations
or mix pre/post-change Nemotron responses into one trial.

Smoke outputs/logs are diagnostic only, under
`results/research_runner/.runtime_smoke/<timestamp>/`. They do not overwrite Item 5
or other completed experiments. Passing one synthetic request per model validates
startup/kernels/JSON output, not every test payload, sustained VRAM use, accuracy
or full-experiment success. The main Audit prompt/schema/settings stay frozen.

Preparation creates the clean venv using the reference Audit Python 3.11
interpreter, not the ML conda Python 3.10; it does not copy/change reference
packages. It needs internet access, disk space, Python >=3.11 for the new venv,
and any required Hugging Face access/license acceptance (`HF_TOKEN` from the
environment, never YAML). The selected vLLM wheel also requires a compatible
NVIDIA driver/CUDA stack. Preparation does not update drivers or stop processes.
The CPU check cannot establish that kernels, JSON decoding and the configured
batch/context fit RTX 4090 VRAM; these still require a real-model run.

Item 5 verifies the saved Llama reference for all five seeds, then checks the
**three new** interpreters/package versions/local snapshots before the first
inference. It stores actual package versions for newly inferred models in
`backend/runtime_preflight.json`. The menu never installs/downloads implicitly.
Use compatible completed Item 2 **Stage 1 and Stage 2** artifacts; do not retrain
or regenerate Screen/Gate scores. Llama has `reuse_configuration: true` and
**must not run again**, regardless of `measure_uncached`. Its exact
Screen-Gate-Audit prefix decisions, metrics, ASR and component timing reports
are copied byte-for-byte from Item 2 for each seed. Saved Audit responses must
match the same configuration/seed/prefix IDs/payloads and Stage 1 routing. Missing
or incompatible source results stop the experiment; there is no inference fallback.

Each Llama output has `reuse.json` with source checksums, zero new inference
requests and explicit inherited timing. `backend/comparison_protocol.json`
distinguishes the inherited reference from three newly measured backends.
The five reused seeds are not five new inference/latency trials. Llama timings
originate from Item 2's mixed-configuration request schedule, so this is not a
fresh, fully matched-workload speed benchmark across all four models.
`measure_uncached: true` applies to the three new candidates only. They run
sequentially in isolated workers, never all loaded on the GPU together.

Gemma is quantized whereas the other candidates are BF16, and serving-engine
versions differ. Report this as a practical **backend/deployment comparison**,
not a controlled experiment attributing differences only to parameter count.
There are no paid/cloud candidates in this profile.

Item 6 reuses Stage 1 Gate test/validation scores and the shared Screen scores.
Each seed's thresholds are selected **only on validation**, using the existing
Stage 1 scores. The four recall floors are 90%, 95%, 98% and 99.9%; for each,
select the highest empirical threshold still meeting the recall floor (strict
score > threshold). This reduces Gate escalation subject to the recall constraint,
rather than maximizing Gate F1. A fifth, separate `validation_max_f1` anchor uses
the original unconstrained max-F1 selector. Score ties/discrete recall may still
make some points coincide; do not invent distinct thresholds or achieved recall.
This sweep-only policy does not change encoder/AE selection or the main per-seed
99.9% constrained-F1 calibration. Both orders use the same threshold and ordered full-test
IDs at each operating point. Screen/Audit models, Screen threshold and Audit
policy stay fixed; new routing determines which Audit calls are needed. Outputs:
`threshold/seed_<seed>/<operating_point>/<order>/`. Shared and per-order
`operating_point.json` record score hashes, calibration and paired-workload hashes.
Audit request/response records identify seed, operating point and order.
At the configured 99.9% point, first try to reuse the matching Item 2 seed/order.
Require exact threshold, current routing/scores, ordered request IDs/payloads,
Audit model/prompt/schema/runtime, source manifests and reconstructed verdicts.
Verified source reports are copied byte-for-byte with zero new Audit requests.
Missing, incomplete, incompatible or nonmatching references cause fresh inference
for that condition, never reuse from another seed/order or silent stale fallback.
Item 2 Stage 2 is optional for Item 6; if unavailable, Item 6 still runs from the
mandatory Stage 1 scores. Setting the reuse list to `[]` disables reference reuse.
Per-condition `execution.json`, `reuse.json` (when reused), aggregate
`execution_summary.csv`/`execution_protocol.json` and summary provenance identify
reused/fresh seeds, fallback reasons and actual new request counts. Inherited
latency is from Item 2's mixed-configuration schedule, not a fresh matched-workload
latency measurement or an independent inference repetition. Do not hide this
distinction in the paper/figure caption. Optional source Stage 2 is hash-linked
as a parent when available; changed source results invalidate resume compatibility.
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
inference repeat. The default uncached timing path does not reuse verdicts,
except the explicitly inherited Item 2 Llama reference in Item 5.
Staged timing is a **component-based pipeline estimate**, not direct end-to-end
latency or production throughput; model-loading time is excluded.

## Live Progress

Every menu item (0-9) prints live `[IG]` updates without extra parameters:
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

## Item 8: Small Auditor-Targeted Case Test (P5#7)

This optional case study runs **one clean prefix and two fixed attack variants**
through `Screen-Gate-Audit`. It does not search for cases or calculate population
F1, bypass rates or episode ASR. Backup is now **Item 9**.

Edit `experiments.auditor_targeted` in `config/experiments.yaml`:

```yaml
seed: 42
case_id: correlated_686_step3  # Prefix ID, not just the episode ID.
order: Screen-Gate-Audit
channel: tool
block_index: 0
json_field: result            # null for plain text; a string field for JSON.
payload_anchor: "The current score baseline is maintained via the 'get_information' tool."
```

Choose the seed/prefix by inspecting Item 2 results. The runner requires a
malicious prefix that Item 2 actually blocked **through Audit**, not Screen.
Choose the channel, zero-based block and exact unique payload anchor manually.
The two template texts live in the same YAML; freeze them before testing.
No system prompt/schema, thresholds, checkpoints or existing data are modified.

Run `python main.py --run 8` (or select 8). The runner:

1. Validates prerequisites and the explicit case, then inserts each template
   immediately after the anchor inside the chosen data block. JSON tool outputs
   remain valid JSON. Both the anchor and full insertion must survive Audit clipping.
2. Copies only that prefix, its episode/reference decision and the three versions
   into `data/auditor_targeted/seed_<seed>/<case_id>/`, with checksums/provenance.
3. Recomputes Screen and Gate scores without training or calibration. The ML
   process exits before Audit starts; Audit runs only for escalated versions.
4. Publishes `results/research_runner/auditor_targeted/REPORT.md`, `comparison.csv`,
   `summary.json`, per-layer decisions/scores and actual Audit requests/responses.

Existing case inputs or results require overwrite confirmation (default No).
A failed rerun leaves the last completed result intact. The copied input directory
is isolated from dataset preparation and training exports. Backup includes the
result reports/config; to also archive the copied inputs, explicitly add
`data/auditor_targeted` to `backup.include_paths`.

The report checks whether the fresh clean control reproduces the original Audit
Stop. If it does not, do not attribute changes solely to injection. An upstream
Stop is a cascade block, not proof that Audit resisted the attack. This is an
offline prefix stress test, not a new live-agent run or an adaptive robustness claim.

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

Item 9 creates a timestamped ZIP in `backup/` using the configured scope. It
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
