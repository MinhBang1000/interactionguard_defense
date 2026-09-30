# ============================================================================
# FILE: utils/utils.py
# General-purpose utility module for preprocessing, reporting, plotting, and I/O.
#
# Purpose:
# - Collects shared helpers used by data preparation, evaluation, plotting,
#   and console-facing workflows.
#
# Workflow:
# - Provides JSON/JSONL read-write helpers.
# - Builds trace sequences and prefix labels.
# - Formats metrics and results for display.
# - Supports figure-generation helper functions.
#
# Use this file when:
# - You need to inspect how traces are turned into sequence samples.
# - You want to understand utility behavior reused across the repository.
# ============================================================================
# utils/utils.py 

import json
import math
import random
import hashlib
import uuid
import re
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from tqdm.auto import tqdm
from models.detectors.pipeline import DetectionPipeline
from utils.flow_analyzer import format_flow_report
from utils.metrics import format_metrics_table
from utils.paths import RESULT_DIR, PROCESSED_DIR
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from datasets import Dataset


def savefig_without_title(save_path, *args, **kwargs):
    """Save plots without titles while preserving legends and axis labels."""
    fig = plt.gcf()
    if getattr(fig, "_suptitle", None) is not None:
        fig._suptitle.set_text("")
    for ax in fig.axes:
        ax.set_title("")
    plt.savefig(save_path, *args, **kwargs)


def make_dataset_for_layer2(data):
    """
    Convert JSONL list into HF Dataset.
    Expected input keys:
      - sequence
      - label
    Output keys:
      - text
      - label
    """
    texts = [item["sequence"] for item in data]
    labels = [int(item["label"]) for item in data]
    return Dataset.from_dict({"text": texts, "label": labels})

def make_dataset(data):

    dataset_dict = {
        "text":[d["sequence"] for d in data],
        "label":[d["label"] for d in data]
    }

    return Dataset.from_dict(dataset_dict)

def load_jsonl(path: Path) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def save_jsonl(data, path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False)+"\n")

def load_json(path: Path) -> Dict:
    data = None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def hash_prompt(text: str) -> str:
    if not text or text is None or str(text).strip() == "":
        text = f"empty_{uuid.uuid4().hex[:8]}"
    return hashlib.md5(str(text).lower().strip().encode()).hexdigest()

def print_split_stats(train, val, test, name: str = "dataset"):
    total = len(train) + len(val) + len(test)
    print(f"\n{name} Split:")
    print(f"   Train: {len(train):>5,} ({len(train)/total*100:>5.1f}%)")
    print(f"   Val:   {len(val):>5,} ({len(val)/total*100:>5.1f}%)")
    print(f"   Test:  {len(test):>5,} ({len(test)/total*100:>5.1f}%)")
    print(f"   Total: {total:>5,}")

def analyze_overlap(data1: List[Dict], data2: List[Dict], name1: str, name2:str) -> int:
    prompts1 = {hash_prompt(t.get("prompt", "")) for t in data1}
    prompts2 = {hash_prompt(t.get("prompt", "")) for t in data2}

    overlap = prompts1 & prompts2

    print(f"\n{name1} & {name2}")
    print(f"   {name1} unique prompts: {len(prompts1):,}")
    print(f"   {name2} unique prompts: {len(prompts2):,}")
    print(f"   Shared prompts: {len(overlap):,}", end="")

    if len(prompts1) > 0:
        pct = len(overlap) / len(prompts1) * 100
        print(f" ({pct:.1f}% of {name1})")
    else:
        print()
    
    return len(overlap)

def add_label(data: List[Dict], label: int, attack_type: str = None) -> List[Dict]:
    for t in data:
        if "label" not in t:
            t["label"] = label
        if "attack_type" not in t:
            t["attack_type"] = attack_type
    return data

def split_data(name: str, data: List[Dict], random_seed: int = 42, train_ratio: int = 0.6, val_ratio: int = 0.2, test_ratio: int = 0.2) -> List[Dict]:
    splitted_data = data
    random.seed(random_seed)
    random.shuffle(splitted_data)
    # get indices
    n_data = len(splitted_data)
    train_end = int(train_ratio*n_data)
    val_end = int(train_end + val_ratio*n_data)
    # split the data
    train_data = splitted_data[:train_end]
    val_data = splitted_data[train_end:val_end]
    test_data = splitted_data[val_end:]

    print_split_stats(train_data, val_data, test_data, name)
    return train_data, val_data, test_data

def normalize_text(text:str) -> str:
    text = text.strip()
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r' +', ' ', text)  # Single spaces only
    return text

def build_trace_sequence(trace: Dict, max_step: Optional[int] = None, special_tokens: Dict = None, add_special_tokens: bool = True) -> str:
    components = []
    SPECIAL_TOKENS = special_tokens
    prompt = trace.get("prompt", "")
    if prompt:
        prompt_text = normalize_text(prompt)
        components.append(f"{SPECIAL_TOKENS['prompt']}\n{prompt_text}")

    reasoning_steps = trace.get("reasoning_steps", [])
    if max_step is not None:
        if max_step == 0:
            reasoning_steps = []
        else:
            reasoning_steps = reasoning_steps[:max_step]
    
    for step in reasoning_steps:
        node = step.get("node", "")
        content = step.get("content", "")
        if not content:
            continue
        content_text = normalize_text(content)
        if node == "retrieve":
            components.append(f"{SPECIAL_TOKENS['retrieve']}\n{content_text}")
        elif node == "agent":
            components.append(f"{SPECIAL_TOKENS['agent']}\n{content_text}")
        elif node.startswith("tool:"):
            tool_name = node.replace("tool:", "") # Only get the name of tool
            tool_token = SPECIAL_TOKENS["tool"].format(name=tool_name)
            components.append(f"{tool_token}\n{content_text}")
        else:
            components.append(f"[{node.upper()}]\n{content_text}")
    
    if add_special_tokens and len(components) > 0:
        sequence = f"\n\n{SPECIAL_TOKENS['sep']}\n\n".join(components)
        sequence += f"\n\n{SPECIAL_TOKENS['eos']}"
    else:
        sequence = "\n\n".join(components)
    
    return sequence

def detect_attack_position(trace: Dict) -> Optional[int]:
    """
        Logic:
            - Prompt: at PROMPT step
            - RAG: attack at the RETRIEVE step
            - Tool: at TOOL step
            - Correlated: at RETRIEVE step (same as RAG)
    """
    if trace["label"] == 0:
        return None
    attack_type = trace.get("attack_type")
    reasoning_steps = trace.get("reasoning_steps", [])

    if attack_type == "prompt":
        return -1
    elif attack_type == "rag":
        for idx, step, in enumerate(reasoning_steps):
            if step.get("node") == "retrieve":
                return idx
        return 0
    elif attack_type == "tool":
        for idx, step in enumerate(reasoning_steps):
            if step.get("node", "").startswith("tool:"):
                return idx
        return 0
    elif attack_type == "correlated":
        for idx, step in enumerate(reasoning_steps):
            if step.get("node") == "retrieve":
                return idx
        
        for idx, step in enumerate(reasoning_steps):
            if step.get("node", "").startswith("tool:"):
                return idx
        return 0
    else:
        return -1
    
def get_label_for_prefix(trace: Dict, max_step: int) -> int:
    original_label = trace["label"]
    if original_label == 0:
        return 0
    attack_pos = detect_attack_position(trace)
    if attack_pos is None:
        return 1
    if attack_pos == -1:
        return 1
    if max_step > 0 and max_step - 1 >= attack_pos:
        return 1
    else:
        return 0
    
def preprocess_trace(
        trace: Dict,
        generate_multi_step: bool = True,
        special_tokens: Dict = None,
        add_special_tokens: bool = True
) -> List[Dict]:
    samples = []
    trace_id = trace.get("id", "unknown")
    original_label = trace["label"]
    attack_type = trace.get("attack_type")
    n_reasoning_steps = len(trace.get("reasoning_steps",[]))

    if generate_multi_step:
        for step_num in range(n_reasoning_steps + 1):
            sequence = build_trace_sequence(
                trace,
                step_num,
                special_tokens,
                add_special_tokens
            )

            prefix_label = get_label_for_prefix(trace, step_num)

            sample = {
                "id": f"{trace_id}_step{step_num}",
                "sequence": sequence,
                "label": prefix_label,
                "attack_type": attack_type,

                # Metadata
                "original_trace_id": trace_id,
                "original_label": original_label,
                "step_number": step_num,
                "total_steps": n_reasoning_steps,
                "is_prefix": step_num < n_reasoning_steps,
                "is_full_trace": step_num == n_reasoning_steps
            }

            samples.append(sample)
    else:
        sequence = build_trace_sequence(
            trace,
            None,
            add_special_tokens
        )

        sample = {
            "id": trace_id,
            "sequence": sequence,
            "label": original_label,
            "attack_type": attack_type,

            "original_trace_id": trace_id,
            "original_label": original_label,
            "step_number": n_reasoning_steps,
            "total_steps": n_reasoning_steps,
            "is_prefix": False,
            "is_full_trace": True
        }

    return samples

def process_file(
    input_path: Path,
    output_path: Path,
    generate_multi_step: bool = True,
    special_tokens: Dict = None,
    verbose: bool = True
) -> Dict[str, any]:
    traces = load_jsonl(input_path)

    all_samples = []
    for trace in tqdm(traces, desc=f"Processing {input_path.name}"):
        samples = preprocess_trace(
            trace=trace,
            generate_multi_step=generate_multi_step,
            special_tokens=special_tokens,
            add_special_tokens=True
        )
        all_samples.extend(samples)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(all_samples, output_path)

    # ===== Stats (Original) =====
    original_labels = Counter(t.get("label", 0) for t in traces)
    original_attack_types = Counter(t.get("attack_type", "none") for t in traces)
    steps_list = [len(t.get("reasoning_steps", [])) for t in traces]
    step_min = min(steps_list) if steps_list else 0
    step_max = max(steps_list) if steps_list else 0
    step_mean = (sum(steps_list) / len(steps_list)) if steps_list else 0.0

    # ===== Stats (Processed) =====
    processed_labels = Counter(s.get("label", 0) for s in all_samples)
    processed_attack_types = Counter(s.get("attack_type", "none") for s in all_samples)

    prefix_cnt = sum(1 for s in all_samples if s.get("is_prefix") is True)
    full_cnt = sum(1 for s in all_samples if s.get("is_full_trace") is True)

    # sanity: among malicious traces, how many prefixes become label=1?
    malicious_trace_ids = {t.get("id") for t in traces if t.get("label", 0) == 1}
    malicious_prefix = [
        s for s in all_samples
        if s.get("original_trace_id") in malicious_trace_ids and s.get("is_prefix") is True
    ]
    mal_prefix_pos = sum(1 for s in malicious_prefix if s.get("label", 0) == 1)
    mal_prefix_total = len(malicious_prefix)
    mal_prefix_rate = (mal_prefix_pos / mal_prefix_total) if mal_prefix_total else 0.0

    stats = {
        "input_file": input_path.name,
        "output_file": output_path.name,
        "original_traces": len(traces),
        "processed_samples": len(all_samples),
        "augmentation_factor": (len(all_samples) / len(traces)) if traces else 0,

        "multi_step_enabled": generate_multi_step,

        "original_distribution": dict(original_labels),
        "processed_distribution": dict(processed_labels),

        "original_attack_type": dict(original_attack_types),
        "processed_attack_type": dict(processed_attack_types),

        "steps_min": step_min,
        "steps_mean": round(step_mean, 2),
        "steps_max": step_max,

        "prefix_samples": prefix_cnt,
        "full_trace_samples": full_cnt,

        "malicious_prefix_positive_rate": round(mal_prefix_rate, 4),
    }

    if verbose:
        print("\n" + "-" * 80)
        print(f"[{input_path.name}]  →  {output_path.name}")
        print(f"multi_step: {generate_multi_step} | traces: {len(traces):,} | samples: {len(all_samples):,} | x{stats['augmentation_factor']:.2f}")
        print(f"original label:  {stats['original_distribution']}")
        print(f"processed label: {stats['processed_distribution']}")
        print(f"attack_type (orig): {stats['original_attack_type']}")
        print(f"steps (min/mean/max): {step_min}/{stats['steps_mean']}/{step_max}")
        if generate_multi_step:
            print(f"prefix/full: {prefix_cnt:,}/{full_cnt:,} | malicious prefix label=1 rate: {stats['malicious_prefix_positive_rate']}")
        print("-" * 80)

    return stats

def save_results(results: Dict, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = maybe_add_cached_layer1_embedding_time(results)

    def convert(obj):
        import numpy as _np
        if isinstance(obj, _np.ndarray):
            return obj.tolist()
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(convert(results), f, indent=2, ensure_ascii=False)
    print(f"\nSaved: {output_path}")

def maybe_add_cached_layer1_embedding_time(results: Dict) -> Dict:
    """
    Post-process pipeline timing to include Layer 1 embedding time when:
    - Layer 1 is used in the mode
    - Layer 1 embeddings were loaded from cache

    The added time is estimated from actual Layer 1 processed samples:
      ceil(layer1_processed / embedding_batch_size) * 3.8
    """
    if results.get("layer1_embedding_time_added"):
        return results

    layer_stats = results.get("layer_stats", {}) or {}
    layer1_stats = layer_stats.get("layer1")
    if not layer1_stats:
        return results

    if not results.get("layer1_embedding_cache_loaded", False):
        return results

    processed = int(layer1_stats.get("processed", 0) or 0)
    if processed <= 0:
        return results

    batch_size = int(results.get("layer1_embedding_batch_size", 128) or 128)
    per_batch_seconds = 3.8
    num_batches = int(math.ceil(processed / batch_size))
    added_seconds = float(num_batches * per_batch_seconds)

    results["time"] = float(results.get("time", 0.0) or 0.0) + added_seconds
    results["layer1_embedding_time_added"] = True
    results["layer1_embedding_time_adjustment"] = {
        "processed_samples": processed,
        "embedding_batch_size": batch_size,
        "num_batches": num_batches,
        "seconds_per_batch": per_batch_seconds,
        "added_seconds": added_seconds,
        "source": "cached_layer1_embeddings",
    }

    print(
        f"Added cached Layer 1 embedding time: +{added_seconds:.2f}s "
        f"({num_batches} batches x {per_batch_seconds:.1f}s) using "
        f"{processed} processed samples."
    )
    return results

def choose_mode_interactively(pipeline: DetectionPipeline) -> int:
    print("\n" + "="*80)
    print("AVAILABLE TEST MODES")

    modes_text = pipeline.list_test_modes()
    print(modes_text)

    while True:
        raw = input("\nEnter test mode ID or 'q' to quit: ").strip().lower()
        if raw in {"q","quit","exit"}:
            raise SystemExit("Bye.")
        try:
            return int(raw)
        except ValueError:
            print("Invalid input, please enter the integer mode ID.")

# def choose_menu_interactively() -> int:
#     print("\n" + "="*60)
#     print("AGENT DEFENSE SYSTEM")
#     print("="*60)
#     print("1) Run multi-layer detection evaluation")
#     print("2) Evaluate prompt injection success (Without Defense - LLM judge)")
#     print("3) Evaluate all other attacks success (Without Defense - Manual judge)")
#     print("4) Evaluate all attacks success (Within Defense)")
#     print("5) Generate all issue figures")
#     print("6) Fine tune ModernBERT for the encoder (layer 1)")
#     print("7) Train the AutoEncoder for the Layer 1 (After fine-tune ModernBERT)")
#     print("8) Fine tune ProtectAI for the Layer 2")
#     print("9) Find the best threshold for the Layer 1")
#     print("10) Run full other baseline test (issue 3)")
#     print("11) Run full other baseline eval (issue 3)")
#     print("12) Run layer 1 threshold tunning -> evaluate layer 3 processed rate (issue 2)")
#     print("0) Exit")
#     print("="*60)

def choose_menu_interactively() -> int:
    print("\n" + "="*60)
    print("AGENT DEFENSE SYSTEM")
    print("="*60)
    print("1) Run multi-layer evaluation")
    print("2) Evaluate prompt attacks (no defense, LLM judge) => NO USE")
    print("3) Evaluate all attacks (no defense, manual)")
    print("4) Evaluate attacks (with 3-layer and baseline defenses)")
    print("5) Generate evaluation figures")
    print("6) Fine-tune Layer 1 encoder (ModernBERT)")
    print("7) Train Layer 1 AutoEncoder")
    print("8) Fine-tune Layer 2 model (ProtectAI)")
    print("9) Find optimal Layer 1 threshold")
    print("10) Run baseline tests (Issue 3)")
    print("11) Evaluate baseline results (Issue 3)")
    print("12) Sweep L1 threshold → analyze L3 load (Issue 2)")
    print("13) Sweep Data Distribution → analyze L3 load (Issue 2)")
    print("0) Exit")
    print("="*60)

    while True:
        choice = input("Select option: ").strip()
        if choice in {"0","1","2","3","4","5","6","7","8","9","10","11","12","13"}:
            return int(choice)
        print("Invalid choice. Please choose again")

def print_results(results: Dict):
    print("\n" + "=" * 80)
    print("📊 EVALUATION RESULTS")
    print("=" * 80)

    if "test_mode" in results:
        print(f"\nTest Mode: {results.get('test_mode')} (ID: {results.get('test_mode_id')})")
    if "strategy" in results:
        print(f"Strategy: {results['strategy']}")

    if "metrics" in results:
        print("\n" + format_metrics_table(results["metrics"]))

    if "layer_stats" in results:
        print("\n" + "=" * 80)
        print("LAYER STATISTICS")
        print("=" * 80)
        for layer, stats in results["layer_stats"].items():
            print(f"\n📌 {layer}")
            for k, v in stats.items():
                if isinstance(v, float):
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")

    if "flow_analysis" in results:
        print("\n" + format_flow_report(results["flow_analysis"]))

    print("\n" + "=" * 80)

def plot_d2_global(mode_result_path: str, out_path: str | None = None):

    p = RESULT_DIR / mode_result_path
    data = json.loads(p.read_text(encoding="utf-8"))

    test_mode = data.get("test_mode", "")
    flow = data.get("flow_analysis", {})
    total_input = flow.get("total_input")
    layers_list = flow.get("layers", [])

    layer_order = test_mode.split("_to_")

    layer_map = {
        d.get("layer"): d
        for d in layers_list
        if isinstance(d, dict) and d.get("layer")
    }

    xs = []
    sent_pct, fwd_pct, blk_pct = [], [], []

    for lname in layer_order:

        d = layer_map.get(lname, {})

        sent = float(d.get("sent_to_next", 0) or 0)
        fwd = float(d.get("forwarded", 0) or 0)
        blk = float(d.get("blocked", 0) or 0)

        xs.append(lname.upper().replace("LAYER", "L"))

        sent_pct.append(sent / total_input * 100)
        fwd_pct.append(fwd / total_input * 100)
        blk_pct.append(blk / total_input * 100)

    totals = [
        sent_pct[i] + fwd_pct[i] + blk_pct[i]
        for i in range(len(xs))
    ]

    if out_path is None:
        out_path = str(p.with_suffix("")) + "_d2_global.png"

    fig = go.Figure()

    # Escalated
    fig.add_bar(
        name="Escalated to Next Layer",
        x=xs,
        y=sent_pct,
        marker=dict(
            color="#4C78A8",
            line=dict(color="black", width=1),
            pattern=dict(shape="/")   # pattern
        )
    )

    # Forwarded
    fig.add_bar(
        name="Forwarded to Agent",
        x=xs,
        y=fwd_pct,
        marker=dict(
            color="#F58518",
            line=dict(color="black", width=1),
            pattern=dict(shape="x")   # pattern
        )
    )

    # Blocked
    fig.add_bar(
        name="Blocked at This Layer",
        x=xs,
        y=blk_pct,
        marker=dict(
            color="#54A24B",
            line=dict(color="black", width=1),
            pattern=dict(shape=".")   # pattern
        )
    )

    # total label
    for i, total in enumerate(totals):
        fig.add_annotation(
            x=xs[i],
            y=total + 2,
            text=f"{total:.0f}%",
            showarrow=False,
            font=dict(size=12)
        )

    fig.update_layout(

        barmode="stack",

        xaxis_title="Layer (Pipeline Order)",
        yaxis_title="Traffic Percentage (%)",

        yaxis=dict(
            range=[0,105],
            ticksuffix="%",
            showgrid=True,
            gridcolor="lightgray"
        ),

        plot_bgcolor="white",

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        ),

        font=dict(
            family="Times New Roman",
            size=14
        )
    )

    fig.write_image(
        RESULT_DIR / out_path,
        width=1100,
        height=550,
        scale=2
    )

    return out_path

def plot_issue1_fpr_tpr(results_dir: str = RESULT_DIR, out_path: str | None = None):
    """
    Issue 1 - Diagram 1: scatter plot (x=FPR, y=TPR) for configurations:
      id=1  -> L1
      id=2  -> L2
      id=11 -> L1-L2
      id=12 -> L1-L2-L3

    JSON must contain: metrics.confusion_matrix = {tn, fp, fn, tp}
    """
    results_dir = Path(results_dir)
    ids = [1, 2, 10, 11, 12]

    id_to_label = {
        1:  "L1",
        2:  "L2",
        10: "L1-L2",
        11: "L1-L2-L3",
        12: "L2-L1-L3",
    }

    colors = {
        "L1":       "#636EFA",
        "L2":       "#EF553B",
        "L1-L2":    "#00CC96",
        "L1-L2-L3": "#AB63FA",
        "L2-L1-L3": "#FFA15A",
    }

    offsets = {
        "L1":       ( 60,   0),
        "L2":       ( 15,  40),
        "L1-L2":    ( 60, -35),
        "L1-L2-L3": ( 60, -35),
        "L2-L1-L3": ( -80,  35),
    }


    def find_file(mode_id: int) -> Path:
        candidates = [
            results_dir / f"{mode_id}_results.json",
            results_dir / f"mode_{mode_id}_results.json",
            results_dir / f"mode_id_{mode_id}_results.json",
            results_dir / f"mode{mode_id}_results.json",
        ]
        for c in candidates:
            if c.exists():
                return c
        hits = sorted(results_dir.glob(f"*{mode_id}*results.json"))
        if hits:
            return hits[0]
        raise FileNotFoundError(f"Cannot find result JSON for mode_id={mode_id} in {results_dir}")

    # ── Load & compute FPR / TPR ─────────────────────────────────────────────
    rows = []
    for mode_id in ids:
        fp = find_file(mode_id)
        data = json.loads(fp.read_text(encoding="utf-8"))

        cm = (data.get("metrics", {}) or {}).get("confusion_matrix", {}) or {}
        tn, fpv, fn, tp = cm.get("tn"), cm.get("fp"), cm.get("fn"), cm.get("tp")
        if None in (tn, fpv, fn, tp):
            raise ValueError(f"Missing tn/fp/fn/tp in metrics.confusion_matrix for {fp.name}")

        tn, fpv, fn, tp = map(int, [tn, fpv, fn, tp])
        fpr = fpv / (fpv + tn) if (fpv + tn) else 0.0
        tpr = tp  / (tp  + fn) if (tp  + fn) else 0.0

        rows.append({
            "mode_id": mode_id,
            "label":   id_to_label.get(mode_id, str(mode_id)),
            "file":    fp.name,
            "FPR":     fpr,
            "TPR":     tpr,
            "TN": tn, "FP": fpv, "FN": fn, "TP": tp,
        })

    df = pd.DataFrame(rows).sort_values("mode_id")
    df.to_csv(results_dir / "issue1_fpr_tpr_points.csv", index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig = go.Figure()

    # One trace per config (separate color + legend entry)
    for _, row in df.iterrows():
        fig.add_trace(go.Scatter(
            x=[row["FPR"]],
            y=[row["TPR"]],
            mode="markers",
            name=row["label"],
            marker=dict(size=14, color=colors.get(row["label"], "#000")),
            hovertemplate=(
                f"{row['label']}<br>"
                f"FPR={row['FPR']:.5f}<br>"
                f"TPR={row['TPR']:.5f}<extra></extra>"
            ),
        ))

    # Annotations with arrows, carefully offset to avoid overlap
    for _, row in df.iterrows():
        ax, ay = offsets.get(row["label"], (15, 15))
        fig.add_annotation(
            x=row["FPR"],
            y=row["TPR"],
            text=(
                f"<b>{row['label']}</b><br>"
                f"FPR={row['FPR']:.4f}<br>"
                f"TPR={row['TPR']:.4f}"
            ),
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=1.5,
            ax=ax, ay=ay,
            font=dict(size=11),
            bgcolor="white",
            bordercolor="#aaa",
            borderwidth=1,
            borderpad=4,
        )

    y_min = max(0.0, float(df["TPR"].min()) - 0.03)
    x_max = max(0.08, float(df["FPR"].max()) * 1.3)

    fig.update_layout(
        xaxis=dict(
            title="False Positive Rate (FPR)",
            range=[0, x_max],
            tickformat=".3f",
            showgrid=True,
            gridcolor="#eee",
        ),
        yaxis=dict(
            title="True Positive Rate (TPR / Recall)",
            range=[y_min, 1.0],    # capped at exactly 1.0
            tickformat=".4f",
            showgrid=True,
            gridcolor="#eee",
        ),
        legend=dict(
            title="Configuration",
            orientation="v",
            x=1.02, y=1,
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    if out_path is None:
        out_path = results_dir / "issue1_fpr_tpr.png"
    else:
        out_path = Path(out_path)
        if not out_path.is_absolute() and out_path.parent == Path("."):
            out_path = results_dir / out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(out_path), width=1000, height=650, scale=2)

    return fig, df, str(out_path)

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_issue1_auc_chart(results_dir: str = RESULT_DIR, out_path: str | None = None):

    results_dir = Path(results_dir)

    ids = [1, 2, 10, 11, 12]

    id_to_label = {
        1:  "L1",
        2:  "L2",
        10: "L1-L2",
        11: "L1-L2-L3",
        12: "L2-L1-L3",
    }

    colors = {
        "L1": "#1f77b4",
        "L2": "#ff7f0e",
        "L1-L2": "#2ca02c",
        "L1-L2-L3": "#d62728",
        "L2-L1-L3": "#9467bd",
    }

    def find_file(mode_id: int) -> Path:
        candidates = [
            results_dir / f"{mode_id}_results.json",
            results_dir / f"mode_{mode_id}_results.json",
            results_dir / f"mode_id_{mode_id}_results.json",
            results_dir / f"mode{mode_id}_results.json",
        ]

        for c in candidates:
            if c.exists():
                return c

        hits = sorted(results_dir.glob(f"*{mode_id}*results.json"))
        if hits:
            return hits[0]

        raise FileNotFoundError(f"Cannot find result JSON for mode_id={mode_id}")

    plt.figure(figsize=(8,6))

    rows = []

    for mode_id in ids:

        fp = find_file(mode_id)
        data = json.loads(fp.read_text())

        roc = data["metrics"]["roc_curve"]

        fpr = np.array(roc["fpr"])
        tpr = np.array(roc["tpr"])

        auc_score = np.trapz(tpr, fpr)

        label = id_to_label[mode_id]

        rows.append({
            "mode_id": mode_id,
            "label": label,
            "AUC": auc_score
        })

        plt.plot(
            fpr,
            tpr,
            lw=2.5,
            color=colors[label],
            label=f"{label} (AUC = {auc_score:.4f})"
        )

    # random baseline
    plt.plot(
        [0,1],
        [0,1],
        linestyle="--",
        color="black",
        lw=2,
        label="Random Guess"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison of Detection Configurations")

    # plt.xlim([0.0, 0.1])   # ⭐ ZOOM AREA
    # plt.ylim([0.9, 1.0])   # ⭐ ZOOM AREA
    
    plt.xlim([0.0, 1.0])   # ⭐ ZOOM AREA
    plt.ylim([0.6, 1.0])   # ⭐ ZOOM AREA

    plt.grid(True, linestyle="--", alpha=0.4)

    plt.legend(loc="lower right")

    if out_path is None:
        out_path = results_dir / "issue1_auc_chart.png"
    else:
        out_path = Path(out_path)

    plt.tight_layout()
    savefig_without_title(out_path, dpi=300)

    df = pd.DataFrame(rows).sort_values("mode_id")
    df.to_csv(results_dir / "issue1_auc_scores.csv", index=False)

    return df, str(out_path)


def is_successful_prompt_injection(prompt_id: str, prompt_failed_path: str = "prompt_injection_failed_cases.json") -> bool:
    prompt_failed_cases = load_json(Path(f"eval/{prompt_failed_path}"))
    for c in prompt_failed_cases:
        if c["id"] == str(prompt_id):
            return False
    return True

def plot_threshold_metrics(data_dir: str,
                           save_path: str = "fig_metrics_vs_threshold.png"):
    """
    Read threshold sweep JSON files and plot 6 key metrics vs threshold.

    Args:
        data_dir (str): directory chứa các file th_*.json
        save_path (str): nơi lưu hình output
    """

    METRICS = ["accuracy", "precision", "recall", "f1_score", "fpr", "fnr"]

    rows = []

    # ===== Load data =====
    for file in sorted(Path(data_dir).glob("th_*.json")):
        match = re.search(r"th_(\d+\.\d+)", file.name)
        if not match:
            continue

        threshold = float(match.group(1))

        with open(file, "r") as f:
            data = json.load(f)

        metrics = data["metrics"]

        row = {"threshold": threshold}
        for m in METRICS:
            row[m] = metrics[m]

        rows.append(row)

    df = pd.DataFrame(rows).sort_values("threshold")

    # ===== Plot =====
    plt.figure()

    for m in METRICS:
        plt.plot(df["threshold"], df[m], marker='o', label=m)

    plt.xlabel("Layer 1 Threshold")
    plt.ylabel("Metric Value")
    plt.title("Layer 1 Threshold Sensitivity Analysis")
    plt.legend()
    plt.grid()

    savefig_without_title(save_path, dpi=300)
    plt.close()

    print(f"Saved: {save_path}")

def couting_attacks(
    data: str = "test_processed.jsonl",
    report_name: str = "report_processed.json"
):
    processed_data = load_jsonl(Path(PROCESSED_DIR / data))

    counting_results = {
        "total_samples": 0,
        "benign": 0,
        "prompt": 0,
        "rag": 0,
        "tool": 0,
        "correlated": 0
    }

    for sample in processed_data:
        counting_results["total_samples"] += 1

        label = sample.get("label", 0)
        
        # ✅ Nếu benign → ignore attack_type
        if label == 0:
            counting_results["benign"] += 1
        else:
            attack_type = sample.get("attack_type", "").lower()
            # ✅ map attack_type → bucket
            if attack_type == "prompt":
                counting_results["prompt"] += 1
            elif attack_type == "rag":
                counting_results["rag"] += 1
            elif attack_type == "tool":
                counting_results["tool"] += 1
            elif attack_type == "correlated":
                counting_results["correlated"] += 1
            else:
                # fallback nếu có type lạ
                print(f"[Warning] Unknown attack type: {attack_type}")

    # -------------------------
    # print report
    # -------------------------
    print("\n===== DATASET STATISTICS =====")
    for k, v in counting_results.items():
        print(f"{k}: {v}")

    # -------------------------
    # save report
    # -------------------------
    with open(Path(PROCESSED_DIR / report_name), "w", encoding="utf-8") as f:
        json.dump(counting_results, f, indent=4)

    return counting_results
