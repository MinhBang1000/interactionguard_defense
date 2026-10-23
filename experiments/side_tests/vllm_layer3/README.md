# Isolated Llama 3.1 8B / vLLM Diagnostic

This side test does not modify menu routing, the main L3 class, .env, Ollama,
or detection_config.yaml. Physical GPU 0 only; GPU 1 remains untouched.
The separate environment follows the [official vLLM GPU installation guidance](https://docs.vllm.ai/en/stable/getting_started/installation/gpu/).

## Environment

Use a separate Python 3.11 virtualenv, not the training environment:

```bash
python3.11 -m venv experiments/side_tests/vllm_layer3/.venv
experiments/side_tests/vllm_layer3/.venv/bin/python -m pip install vllm==0.11.2
```

The installed environment's exact resolved packages are frozen alongside the
result. The original cached HF Llama-3.1-8B-Instruct snapshot is pinned; there
is no model fine-tuning, quantization or download of replacement weights.

## Prepare and Run

From the repository root, using conda agent_defense to prepare:

```bash
python -m experiments.side_tests.vllm_layer3.prepare
```

This prints a new result directory under results/side_vllm_layer3. It freezes
the current successful saved L1/L2 scores, original/cleaned prompt and prompt
diff, and O2-formatted inputs. Labels and scenario metadata are stored for
scoring only, never passed to the LLM. Then use the isolated interpreter:

```bash
experiments/side_tests/vllm_layer3/.venv/bin/python -m experiments.side_tests.vllm_layer3.run RESULTS_DIRECTORY
```

The runner checks that physical GPU 0 is idle and sets CUDA_VISIBLE_DEVICES=0
before importing Torch/vLLM. vLLM runs as an offline batched engine, not a
public HTTP service. Settings: BF16, tensor parallel 1, 8192-token context,
max 8 active sequences, chunked prefill with 4096 batched tokens, prefix cache,
88% GPU memory budget, temperature 0, seed 42 and JSON-schema constrained output
up to 512 tokens. These are conservative hardware settings, not a demonstrated
globally optimal serving configuration.

After completion, verify all saved predictions and export summary.csv:

```bash
python -m experiments.side_tests.vllm_layer3.summarize RESULTS_DIRECTORY
```

## Protocol and Results

Only L1-suspicious / L2-negative prefixes reach L3. Process thresholds in order
0.25715, 0.20380, 0.10235, 0.07753. Shared identical payloads reuse deterministic
L3 results; each threshold writes its full 5445-prefix cascade metrics and
per-prefix predictions immediately after completion. Watch runner stdout for
COMPLETED, or read summary.json and threshold_*.json. L3 subset confusion is
reported separately from full cascade confusion. Invalid/truncated responses
raise errors; they are never silently scored as benign.

L3 blocks when any malicious branch/final verdict is true AND confidence >=0.5,
matching O2's decision rule. L1 bypassed positives remain missed and L2 blocked
negatives cannot be undone. Output includes all raw JSON verdicts and evidence.

This is exploratory: the test was inspected before this run. The cleaned
prompt removes the explicit malicious examples, verdict-bearing JSON example
and target-tool identifiers, but retains O2's authorization and attack-presence
policy. Current O2 character/block clipping is retained, not replaced with an
unlimited full-history input. Backend/dtype, output constraints and token cap
also differ from the earlier setup. Do not describe this as an accuracy-neutral
backend-only change, as a new untouched confirmation, or as evidence that all
possible leakage has been independently ruled out.

Incremental wall times include caching/batch effects and are not comparable
end-to-end per-sample deployment latency. No ASR is calculated. This does not
install a new backend into the main workflow or change its active thresholds.

## Optional Stored-Trace ASR Scoring

```bash
python -m experiments.side_tests.vllm_layer3.score_asr RESULTS_DIRECTORY
```

This separate post-processing step uses eval/asr.py unchanged. It wraps the
side-test prediction IDs with the verified processed-data hash and maps them
back to original episodes. A block counts as prevention only before the
contract's last-safe-prefix boundary, based on the conservative tool-result
timing proxy. Naturally failed attacks are not credited as prevented.
It creates a fresh asr/ directory and refuses overwriting an existing one.

Current prompt-injection episodes lack a valid success-judge artifact, so their
ASR remains UNKNOWN. Three-type pooled ASR includes only RAG/tool/correlated
attempts, not a silent complete-case substitute for a four-category result.
All-category bounds and benign episode false-block rate are also reported.
This is offline counterfactual replay, not a new live agent execution.
