# ============================================================================
# FILE: main.py
# Interactive command console for the entire Agent Defense research workflow.
#
# Purpose:
# - Acts as the primary user-facing entry point for this repository.
# - Organizes the project into terminal menus for defense runs, baselines,
#   evaluations, figure generation, training, sweeps, and documentation.
#
# Workflow:
# - Users start here with `python main.py`.
# - The file renders the startup brand screen, then exposes grouped menus.
# - Each menu dispatches to an existing experiment or evaluation function.
# - Guided workflows chain multiple steps together for common research tasks.
#
# Use this file when:
# - You want to operate the project from the terminal without manually calling
#   individual scripts.
# - You want to understand the intended order of execution across pipeline,
#   baseline, evaluation, and figure-generation stages.
# ============================================================================

from utils.utils import load_jsonl, save_results, choose_mode_interactively, print_results
from results.issue_figure_generation import generate_full_figures
from utils.paths import RESULT_DIR, CONFIG_DIR, PROCESSED_DIR
from experiments.common import (
    normalize_layer3_variant as _normalize_layer3_variant,
    read_json as _read_json,
    write_json as _write_json,
    write_jsonl as _write_jsonl,
)
from config.config import Issue2HighBenignSweepConfig
from eval.eval_asr_without_defense_pi import prompt_injection_eval
from eval.eval_asr_without_defense import filter_attack_type
from eval.eval_asr_within_defense import eval_with_defense, eval_with_baseline
import numpy as np
import pandas as pd
import shutil
import subprocess
import sys
import os
import random
import yaml
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[96m"
BLUE = "\033[94m"
WHITE = "\033[97m"
GOLD = "\033[93m"
RED = "\033[91m"
GREEN = "\033[92m"
MAGENTA = "\033[95m"


def print_startup_brand():
    print(
        f"""
{CYAN}{BOLD}    █████╗  ██████╗ ███████╗███╗   ██╗████████╗
   ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝
   ███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║
   ██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║
   ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║
   ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝{RESET}

{BLUE}{BOLD}    ██████╗ ███████╗███████╗███████╗███╗   ██╗███████╗███████╗
    ██╔══██╗██╔════╝██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
    ██║  ██║█████╗  █████╗  █████╗  ██╔██╗ ██║███████╗█████╗
    ██║  ██║██╔══╝  ██╔══╝  ██╔══╝  ██║╚██╗██║╚════██║██╔══╝
    ██████╔╝███████╗██║     ███████╗██║ ╚████║███████║███████╗
    ╚═════╝ ╚══════╝╚═╝     ╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝{RESET}

{WHITE}{BOLD}    3-LAYER SECURITY FOR LLM AGENTS{RESET}
{DIM}    Research Console | Detection | Baselines | Evaluation | Figures{RESET}

{GOLD}{BOLD}    Le Minh Bang{RESET}
{RED}{BOLD}    High Speed Network Lab | NYCU | Taiwan{RESET}
"""
    )


def print_header(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def ask_choice(title, options):
    print_header(title)
    for key, label in options:
        print(f"{key}) {label}")

    valid = {str(key) for key, _ in options}
    while True:
        choice = input("Select option: ").strip()
        if choice in valid:
            return choice
        print("Invalid choice. Please choose one of:", ", ".join(valid))


def ask_mode(prompt="Enter mode (11 or 12): ", allowed=("11", "12")):
    while True:
        mode = input(prompt).strip()
        if mode in allowed:
            return int(mode)
        print(f"Invalid input. Please enter one of: {', '.join(allowed)}")


def ask_percentage(prompt="Enter train percentage (1-100): "):
    while True:
        raw = input(prompt).strip()
        try:
            value = int(raw)
        except ValueError:
            print("Invalid input. Please enter an integer from 1 to 100.")
            continue

        if 1 <= value <= 100:
            return value
        print("Invalid input. Please enter an integer from 1 to 100.")

def prepare_test_data_for_mode(test_data, mode_id):
    if mode_id not in {3, 11, 12, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44}:
        return test_data

    print("\nPipeline test subset")
    print("Choose how much of test_processed.jsonl to use as the initial input for this run.")
    print("Layer routing remains unchanged; Layer 1/2 still decide which samples reach Layer 3.")
    print("If the ratio is below 100%, the test set will be shuffled before sampling.\n")

    ratio = ask_percentage("Enter test percentage for this pipeline run (1-100): ")
    if ratio == 100:
        print(f"Using full test set: {len(test_data)} / {len(test_data)} samples\n")
        return test_data

    shuffled = list(test_data)
    np.random.shuffle(shuffled)
    subset_size = max(1, int(len(shuffled) * (ratio / 100.0)))
    subset = shuffled[:subset_size]

    print(f"Using shuffled test subset: {subset_size} / {len(test_data)} samples ({ratio}%)\n")
    return subset


def ask_with_default(prompt, default):
    value = input(prompt).strip()
    return value or default


def ask_yes_no(prompt, default=False):
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return bool(default)
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def mode_uses_selectable_layer3(mode_id):
    test_modes_path = CONFIG_DIR / "test_modes.yaml"
    with open(test_modes_path, "r", encoding="utf-8") as f:
        modes_cfg = yaml.safe_load(f) or {}
    mode_cfg = modes_cfg.get("test_modes", {}).get(int(mode_id), {})
    return "layer3" in mode_cfg.get("layers", [])


def ask_layer3_variant_for_mode(mode_id):
    if not mode_uses_selectable_layer3(mode_id):
        return None

    with open(CONFIG_DIR / "detection_config.yaml", "r", encoding="utf-8") as f:
        detection_cfg = yaml.safe_load(f) or {}
    configured_layers = set((detection_cfg.get("layers") or {}).keys())

    print("\nLayer 3 version")
    print("This mode uses logical layer3. Choose which implementation should back it.")
    print("Press Enter for default/current behavior from detection_config.yaml.\n")
    print("default) current logical layer3")
    print("orig)    Original Layer3Detector")
    print("o2)      Layer3DetectorO2")
    print("custom)  Type a configured key such as layer3o2")

    while True:
        raw = input("Select Layer 3 version [default]: ").strip().lower()
        if raw in {"", "default"}:
            print("Using default logical layer3.\n")
            return None
        if raw in {"o2", "layer3o2", "single", "channel", "channel-audit", "channelaudit"}:
            variant = "layer3o2"
        elif raw in {"1", "v1", "orig", "original", "legacy", "layer3orig"}:
            variant = "layer3orig"
        elif raw.startswith("layer3"):
            variant = raw
        else:
            print("Invalid choice. Please choose default, orig, o2, or a configured key like layer3o2.")
            continue

        if variant not in configured_layers:
            print(f"{variant} is not configured in detection_config.yaml.")
            continue

        print(f"Using {variant} for logical layer3.\n")
        return variant


def layer3_variant_suffix(layer3_variant):
    if not layer3_variant:
        return ""
    if str(layer3_variant).startswith("layer3"):
        return f"_{layer3_variant}"
    return f"_layer3{layer3_variant}"


def run_isolation_layer1_sweep():
    print_header("ISOLATED LAYER 1 RETRAIN + THRESHOLD SWEEP")
    print("This runs an isolated experiment only.")
    print("It evaluates full Mode 11: Layer 1 -> Layer 2 -> Layer 3.")
    print("It writes artifacts under results/isolation_layer1_sweep/<run_id>/")
    print("Original model directories and project configs are not modified.\n")

    train_percents = ask_with_default(
        "Train percentages [default: 50,75,100]: ",
        "50,75,100",
    )
    run_id = ask_with_default(
        "Run id [default: timestamp generated by script]: ",
        "",
    )
    encoder_epochs = ask_with_default(
        "Encoder epochs [default: script default]: ",
        "",
    )
    ae_epochs = ask_with_default(
        "AutoEncoder epochs [default: script default]: ",
        "",
    )

    thresholds = ",".join(f"{value / 100:.2f}" for value in range(10, 21))
    command = [
        sys.executable,
        "scripts/isolation_layer1_sweep.py",
        "--train_percents",
        train_percents,
        "--thresholds",
        thresholds,
    ]
    if run_id:
        command.extend(["--run_id", run_id])
    if encoder_epochs:
        command.extend(["--encoder_epochs", encoder_epochs])
    if ae_epochs:
        command.extend(["--ae_epochs", ae_epochs])

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_DEVICE_ML", "0"))

    print("\nRunning:")
    print(" ".join(command))
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    print("Threshold sweep fixed to 0.10 -> 0.20.\n")

    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


def run_extract_isolation_fp_samples():
    print_header("EXTRACT ISOLATION FALSE POSITIVES")
    print("This reads a completed isolation Mode 11 result and writes FP samples to JSONL.")
    print("Leave run id empty to use the latest isolation run.")
    print("Leave threshold empty to use train_pct best_result.json.\n")

    run_id = ask_with_default("Run id [default: latest]: ", "")
    train_percent = ask_with_default("Train percent [default: 50]: ", "50")
    threshold = ask_with_default("Threshold [default: best for train percent]: ", "")
    output = ask_with_default("Output JSONL [default: inside train_pct dir]: ", "")

    command = [
        sys.executable,
        "scripts/extract_isolation_fp_samples.py",
        "--train_percent",
        train_percent,
    ]
    if run_id:
        command.extend(["--run_id", run_id])
    if threshold:
        command.extend(["--threshold", threshold])
    if output:
        command.extend(["--output", output])

    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


def run_reevaluate_isolation_models():
    print_header("RE-EVALUATE TRAINED ISOLATION MODELS")
    print("This reuses trained ModernBERT/AE artifacts and reruns full Mode 11.")
    print("Use it for Layer 3 prompt ablations without retraining Layer 1.\n")

    run_id = ask_with_default("Run id [default: latest completed]: ", "")
    train_percents = ask_with_default("Train percents [default: 50]: ", "50")
    thresholds = ask_with_default("Thresholds [default: best | all | comma list]: ", "best")
    output_name = ask_with_default("Output folder name [default: timestamp]: ", "")

    command = [
        sys.executable,
        "scripts/reevaluate_isolation_models.py",
        "--train_percents",
        train_percents,
        "--thresholds",
        thresholds,
    ]
    if run_id:
        command.extend(["--run_id", run_id])
    if output_name:
        command.extend(["--output_name", output_name])

    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", os.getenv("CUDA_DEVICE_ML", "0"))

    print("\nRunning:")
    print(" ".join(command))
    print(f"CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    subprocess.run(command, cwd=Path(__file__).resolve().parent, env=env, check=True)


def run_backup_current_results():
    print_header("BACKUP CURRENT RESULTS")
    print("This creates a timestamped zip under backups/ with a file-level progress bar.")
    print("It saves source code, configs, data artifacts, eval files, docs, and results.")
    print("Git internals, nested backups, .env, caches, and large model/checkpoint files are excluded.\n")

    label = ask_with_default(
        "Backup label [default: current_results]: ",
        "current_results",
    )
    command = [
        "scripts/backup_current_results.sh",
        label,
    ]

    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, cwd=Path(__file__).resolve().parent, check=True)


def h1(text):
    return f"{CYAN}{BOLD}{text}{RESET}"


def h2(text):
    return f"{BLUE}{BOLD}{text}{RESET}"


def accent(text):
    return f"{GOLD}{BOLD}{text}{RESET}"


def note(text):
    return f"{DIM}{text}{RESET}"


def bullet(text):
    return f"{GREEN}•{RESET} {text}"


def build_intro_document():
    return [
        h1("AGENT DEFENSE | INTRODUCTION"),
        "",
        h2("The Problem"),
        "Modern LLM agents do not only read user prompts.",
        "They also consume retrieved documents, tool results, and multi-step context.",
        "That makes them more useful, but it also makes them easier to manipulate.",
        "",
        bullet("A malicious instruction can hide inside a retrieved document."),
        bullet("A harmful tool result can redirect the agent away from the user goal."),
        bullet("A correlated attack can spread influence across multiple channels."),
        "",
        "The real failure is not only bad text generation.",
        "The deeper failure is behavioral drift: the agent still sounds fluent,",
        "but it has already stopped serving the original user intent.",
        "",
        h2("The Solution"),
        "Agent Defense addresses this problem with a 3-layer design.",
        "",
        bullet(f"{accent('Layer 1')} screens for unfamiliar traces using anomaly detection."),
        bullet(f"{accent('Layer 2')} blocks patterns that resemble known malicious behavior."),
        bullet(f"{accent('Layer 3')} reasons semantically over the trace and audits intent alignment."),
        "",
        h2("Why 3 Layers"),
        "No single detector is enough for all attack classes.",
        "The layered design lets the system balance speed, cost, and semantic coverage.",
        "",
        bullet("Layer 1 reduces downstream load."),
        bullet("Layer 2 catches known attack-like patterns efficiently."),
        bullet("Layer 3 handles harder semantic manipulation cases."),
        "",
        h2("Research Value"),
        "This repository is both an implementation and an experimental platform.",
        "It supports data processing, model training, baseline comparison, ASR evaluation,",
        "threshold sweeps, distribution sweeps, and paper-ready figure generation.",
        "",
        note("Think of this project as a security systems lab for LLM agents, not only a classifier benchmark."),
    ]


def build_workflow_document():
    return [
        h1("AGENT DEFENSE | WORKFLOW GUIDE"),
        "",
        h2("Best Entry Points"),
        bullet("Defense pipeline: run direct mode evaluation"),
        bullet("Baselines: refresh RAGuard, Tool Result Parsing, Prompt Sentinel, Combined"),
        bullet("Evaluations: refresh ASR reports"),
        bullet("Guided workflows: fastest way to complete common tasks"),
        "",
        h2("Most Useful Paths"),
        bullet(f"{accent('Full 3-layer result')}: 1 -> 3"),
        bullet(f"{accent('Run Tool Result Parsing baseline')}: 2 -> 2"),
        bullet(f"{accent('Build Combined baseline')}: 2 -> 4"),
        bullet(f"{accent('Refresh baseline evaluation')}: 3 -> 4"),
        bullet(f"{accent('Refresh defense + baseline evaluation')}: 3 -> 3"),
        bullet(f"{accent('Generate all figures')}: 4 -> 1"),
        bullet(f"{accent('Fine-tune local Llama 3.1 for Layer 3')}: 5 -> 4"),
        bullet(f"{accent('Deploy a fine-tuned Layer 3 model to Ollama')}: 5 -> 5"),
        "",
        h2("Issue 3 ASR Line Chart"),
        "If the outputs are already prepared:",
        bullet("7 -> 4"),
        "",
        "If you want the full workflow from scratch:",
        bullet("7 -> 5"),
        "",
        h2("Important Dependency Reminder"),
        "The Issue 3 line chart does not read raw baseline metrics directly.",
        "It depends on evaluation reports, so after prediction changes,",
        "you must refresh evaluation before regenerating figures.",
        "",
        note("Use Guided Workflows when you want reliability. Use the lower-level menus when you want control."),
    ]


def build_about_document():
    return [
        h1("ABOUT LE MINH BANG"),
        "",
        f"{GOLD}{BOLD}Le Minh Bang{RESET}",
        f"{RED}{BOLD}High Speed Network Lab | NYCU | Taiwan{RESET}",
        "",
        h2("Research Identity"),
        "This project presents you as a researcher who treats LLM security as a systems problem.",
        "That is an important distinction. The work is not framed as a narrow prompt filter,",
        "but as a defense architecture for agents operating across multiple channels.",
        "",
        h2("What This Project Says About You"),
        bullet("You care about trustworthy agent behavior, not only model accuracy."),
        bullet("You are comfortable combining engineering implementation with research evaluation."),
        bullet("You think in terms of layered control, not one-model shortcuts."),
        bullet("You build artifacts that can support both experiments and storytelling."),
        "",
        h2("Research Narrative"),
        "As LLM agents become more connected to tools and external knowledge,",
        "their attack surface expands. Agent Defense is your answer to that shift.",
        "It proposes a 3-layer architecture that screens, classifies, and reasons.",
        "",
        h2("Why This Matters"),
        "A strong research project does more than report a score.",
        "It explains the problem, proposes a credible solution, and provides a workflow for testing it.",
        "This repository already does that.",
        "",
        note("In short: this is the profile of a builder-researcher working seriously on robust LLM agent systems."),
    ]


def build_quick_reference_document():
    return [
        h1("AGENT DEFENSE | QUICK REFERENCE"),
        "",
        h2("Fast Recipes"),
        bullet("Run full 3-layer pipeline: 1 -> 3"),
        bullet("Run all baselines: 7 -> 2"),
        bullet("Refresh baseline reports only: 7 -> 3"),
        bullet("Issue 3 chart from ready outputs: 7 -> 4"),
        bullet("Issue 3 chart from scratch: 7 -> 5"),
        bullet("Fine-tune local Llama 3.1: 5 -> 4"),
        bullet("Deploy fine-tuned Llama 3.1 to Ollama: 5 -> 5"),
        bullet("Threshold sweep: 6 -> 1"),
        bullet("Distribution sweep: 6 -> 2"),
        "",
        h2("Mental Model"),
        bullet("Menu 1 runs pipeline outputs."),
        bullet("Menu 2 builds baseline predictions."),
        bullet("Menu 3 builds evaluation reports."),
        bullet("Menu 4 generates figures."),
        bullet("Menu 7 chains steps together for you."),
        "",
        note("If you are unsure, start with Guided Workflows."),
    ]


def view_document_lines(lines, title):
    term_height = shutil.get_terminal_size(fallback=(100, 30)).lines
    page_size = max(10, term_height - 6)
    index = 0

    while True:
        print_header(title)
        page = lines[index:index + page_size]
        if page:
            print("\n".join(page))
        else:
            print("[Empty document]")

        if index + page_size >= len(lines):
            print(f"\n{DIM}End of document{RESET}")
            cmd = input("Press q to return, or b to go back one page: ").strip().lower()
            if cmd == "q":
                return
            if cmd == "b":
                index = max(0, index - page_size)
            continue

        cmd = input("Press Enter for next page, b for back, q to return: ").strip().lower()
        if cmd == "q":
            return
        if cmd == "b":
            index = max(0, index - page_size)
        else:
            index += page_size

def run_detection_pipeline():
    test_path = PROCESSED_DIR / "test_processed.jsonl"
    config_path = CONFIG_DIR / "detection_config.yaml"
    out_dir = RESULT_DIR
    test_data = load_jsonl(test_path)
    if not test_data:
        raise ValueError("Empty test-data JSONL")

    from models.detectors.pipeline import DetectionPipeline
    pipeline = DetectionPipeline(config_path=config_path, test_mode=None)
    mode_id = choose_mode_interactively(pipeline=pipeline)
    layer3_variant = ask_layer3_variant_for_mode(mode_id)
    pipeline = DetectionPipeline(
        config_path=config_path,
        test_mode=mode_id,
        layer3_variant=layer3_variant,
    )
    test_data = prepare_test_data_for_mode(test_data, mode_id)

    sequences = [d["sequence"] for d in test_data]
    labels = np.array([d["label"] for d in test_data])
    results = pipeline.predict(
        sequences=sequences,
        labels=labels,
        return_details=True,
    )
    print_results(results)
    save_results(results, out_dir / f"mode_{mode_id}{layer3_variant_suffix(layer3_variant)}_results.json")


def run_detection_pipeline_for_mode(mode_id, layer3_variant=None, prompt_layer3_variant=True):
    test_path = PROCESSED_DIR / "test_processed.jsonl"
    config_path = CONFIG_DIR / "detection_config.yaml"
    out_dir = RESULT_DIR
    test_data = load_jsonl(test_path)
    if not test_data:
        raise ValueError("Empty test-data JSONL")
    test_data = prepare_test_data_for_mode(test_data, mode_id)
    if prompt_layer3_variant and layer3_variant is None:
        layer3_variant = ask_layer3_variant_for_mode(mode_id)

    from models.detectors.pipeline import DetectionPipeline
    pipeline = DetectionPipeline(
        config_path=config_path,
        test_mode=mode_id,
        layer3_variant=layer3_variant,
    )

    sequences = [d["sequence"] for d in test_data]
    labels = np.array([d["label"] for d in test_data])
    results = pipeline.predict(
        sequences=sequences,
        labels=labels,
        return_details=True,
    )
    print_results(results)
    save_results(results, out_dir / f"mode_{mode_id}{layer3_variant_suffix(layer3_variant)}_results.json")

def run_distribution_sweep(mode_id=11, fixed_threshold=0.14, repeats=3, layer3_variant=None):
    print("\n🚀 RUNNING DISTRIBUTION SWEEP\n")

    import random
    import numpy as np
    import pandas as pd

    test_path = PROCESSED_DIR / "test_processed.jsonl"
    config_path = CONFIG_DIR / "detection_config.yaml"

    test_data = load_jsonl(test_path)
    if not test_data:
        raise ValueError("Empty test-data")
    if layer3_variant is None:
        layer3_variant = ask_layer3_variant_for_mode(mode_id)

    benign_data = [d for d in test_data if d["label"] == 0]
    attack_types = ["prompt", "rag", "tool", "correlated"]
    malicious_dict = {
        atk: [d for d in test_data if d["label"] == 1 and d["attack_type"] == atk]
        for atk in attack_types
    }

    total_size = 1640
    sweep_dir = RESULT_DIR / "distribution_sweep" / f"mode_{mode_id}{layer3_variant_suffix(layer3_variant)}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    summary = []

    from models.detectors.pipeline import DetectionPipeline

    ratios = np.arange(0.1, 0.91, 0.1)

    for ratio in ratios:
        print(f"\n[Benign Ratio = {ratio:.1f}]")

        for r in range(repeats):
            print(f"  -> Repeat {r+1}/{repeats}")

            n_benign = int(total_size * ratio)
            n_mal_total = total_size - n_benign
            n_per_type = n_mal_total // len(attack_types)
            sampled = []
            sampled += random.sample(benign_data, n_benign)
            for atk in attack_types:
                sampled += random.sample(malicious_dict[atk], n_per_type)

            random.shuffle(sampled)

            sequences = [d["sequence"] for d in sampled]
            labels = np.array([d["label"] for d in sampled])

            pipeline = DetectionPipeline(
                config_path=config_path,
                test_mode=mode_id,
                layer3_variant=layer3_variant,
            )
            pipeline.layers["layer1"].threshold = float(fixed_threshold)

            results = pipeline.predict(
                sequences=sequences,
                labels=labels,
                return_details=True,
            )

            out_path = sweep_dir / f"benign_{int(ratio*100)}_run{r+1}.json"
            results["sweep_metadata"] = {
                "sweep_type": "distribution",
                "mode": mode_id,
                "layer3_variant": layer3_variant or "default",
                "selected_layer3_variant": results.get("selected_layer3_variant", "layer3"),
            }
            save_results(results, out_path)

            layer3_processed = results["layer_stats"].get("layer3", {}).get("processed", 0)
            summary.append({
                "ratio": ratio,
                "run": r,
                "L3_ratio": layer3_processed / len(sequences),
                "time": results["time"],
                "precision": results["metrics"]["precision"],
                "recall": results["metrics"]["recall"],
                "f1": results["metrics"]["f1_score"],
                "fpr": results["metrics"]["fpr"],
                "fnr": results["metrics"]["fnr"]
            })

    df = pd.DataFrame(summary)
    df.to_csv(sweep_dir / "summary.csv", index=False)


def _sample_high_benign_subset(test_data, total_size, benign_ratio, rng):
    benign_indices = [i for i, d in enumerate(test_data) if d["label"] == 0]
    malicious_indices_by_type = {
        atk: [
            i for i, d in enumerate(test_data)
            if d["label"] == 1 and d.get("attack_type") == atk
        ]
        for atk in Issue2HighBenignSweepConfig.ATTACK_TYPES
    }

    n_benign = int(total_size * benign_ratio)
    n_mal_total = total_size - n_benign

    if n_benign > len(benign_indices):
        raise ValueError(
            f"Need {n_benign} benign samples for ratio {benign_ratio:.2f}, "
            f"but only {len(benign_indices)} are available."
        )

    selected_benign = rng.sample(benign_indices, n_benign)

    n_attack_types = len(Issue2HighBenignSweepConfig.ATTACK_TYPES)
    balanced_take = n_mal_total // n_attack_types
    selected_malicious = []
    remaining_by_type = {}

    for attack_type in Issue2HighBenignSweepConfig.ATTACK_TYPES:
        pool = malicious_indices_by_type[attack_type]
        take = min(balanced_take, len(pool))
        chosen = rng.sample(pool, take)
        selected_malicious.extend(chosen)
        chosen_set = set(chosen)
        remaining_by_type[attack_type] = [idx for idx in pool if idx not in chosen_set]

    remaining_needed = n_mal_total - len(selected_malicious)
    if remaining_needed > 0:
        remaining_pool = []
        for attack_type in Issue2HighBenignSweepConfig.ATTACK_TYPES:
            remaining_pool.extend(remaining_by_type[attack_type])

        if remaining_needed > len(remaining_pool):
            raise ValueError(
                f"Need {remaining_needed} extra malicious samples after balanced sampling, "
                f"but only {len(remaining_pool)} remain."
            )

        selected_malicious.extend(rng.sample(remaining_pool, remaining_needed))

    sampled_indices = selected_benign + selected_malicious
    rng.shuffle(sampled_indices)
    sampled = [test_data[i] for i in sampled_indices]

    per_attack_counts = {}
    for attack_type in Issue2HighBenignSweepConfig.ATTACK_TYPES:
        per_attack_counts[attack_type] = sum(
            1 for i in selected_malicious if test_data[i].get("attack_type") == attack_type
        )

    return sampled, {
        "benign_count": n_benign,
        "malicious_count": n_mal_total,
        "actual_benign_ratio": n_benign / total_size,
        "attack_type_counts": per_attack_counts,
    }


def _load_layer1_threshold_from_detection_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f) or {}
    return float(
        config_data.get("layers", {})
        .get("layer1", {})
        .get("params", {})
        .get("threshold", 0.14)
    )


def run_high_benign_distribution_sweep():
    print("\n🚀 RUNNING HIGH-BENIGN DISTRIBUTION SWEEP\n")

    cfg = Issue2HighBenignSweepConfig
    repeats_default = str(cfg.REPEATS)
    start_default = str(cfg.BENIGN_START_PCT)
    end_default = str(cfg.BENIGN_END_PCT)
    step_default = str(cfg.BENIGN_STEP_PCT)

    def _ask_positive_int(prompt, default):
        while True:
            raw = ask_with_default(prompt, default)
            try:
                value = int(raw)
            except ValueError:
                print("Invalid input. Please enter a positive integer.")
                continue
            if value >= 1:
                return value
            print("Invalid input. Please enter a positive integer.")

    def _ask_pct(prompt, default):
        while True:
            raw = ask_with_default(prompt, default)
            try:
                value = int(raw)
            except ValueError:
                print("Invalid input. Please enter an integer percentage from 1 to 99.")
                continue
            if 1 <= value <= 99:
                return value
            print("Invalid input. Please enter an integer percentage from 1 to 99.")

    mode_id = ask_mode("Enter mode for high-benign sweep (11 or 12): ")
    layer3_variant = ask_layer3_variant_for_mode(mode_id)
    repeats = _ask_positive_int(f"Enter repeats [default: {repeats_default}]: ", repeats_default)
    benign_start_pct = _ask_pct(
        f"Enter benign start percentage [default: {start_default}]: ",
        start_default,
    )
    benign_end_pct = _ask_pct(
        f"Enter benign end percentage [default: {end_default}]: ",
        end_default,
    )
    benign_step_pct = _ask_positive_int(
        f"Enter benign step percentage [default: {step_default}]: ",
        step_default,
    )

    if benign_start_pct > benign_end_pct:
        raise ValueError("Benign start percentage must be <= benign end percentage.")

    test_path = PROCESSED_DIR / "test_processed.jsonl"
    config_path = CONFIG_DIR / "detection_config.yaml"
    test_data = load_jsonl(test_path)
    if not test_data:
        raise ValueError("Empty test-data")

    total_size = cfg.FIXED_TOTAL_SIZE
    threshold = _load_layer1_threshold_from_detection_config(config_path)
    benign_available = sum(1 for d in test_data if d["label"] == 0)
    malicious_available = sum(1 for d in test_data if d["label"] == 1)

    print(f"Fixed sample size         : {total_size}")
    print(f"Available benign          : {benign_available}")
    print(f"Available malicious       : {malicious_available}")
    print(f"Benign range              : {benign_start_pct}% -> {benign_end_pct}% (step {benign_step_pct}%)")
    print(f"Mode                      : {mode_id}")
    print(f"Layer 3 variant           : {layer3_variant or 'default'}")
    print(f"Repeats                   : {repeats}")
    print(f"Layer 1 threshold         : {threshold} (read from detection_config.yaml)")
    print("Malicious sampling policy : balanced across attack types first, then random fill\n")

    sweep_root = RESULT_DIR / cfg.OUTPUT_SUBDIR
    sweep_root.mkdir(parents=True, exist_ok=True)
    mode_dir = sweep_root / f"mode_{mode_id}{layer3_variant_suffix(layer3_variant)}"
    mode_dir.mkdir(parents=True, exist_ok=True)

    sweep_manifest = {
        "fixed_total_size": total_size,
        "repeats": repeats,
        "benign_start_pct": benign_start_pct,
        "benign_end_pct": benign_end_pct,
        "benign_step_pct": benign_step_pct,
        "mode": mode_id,
        "layer3_variant": layer3_variant or "default",
        "layer1_threshold_from_config": threshold,
        "sampling_policy": "balanced_across_attack_types_then_random_fill",
        "max_supported_fixed_total_size_for_99pct_benign": cfg.FIXED_TOTAL_SIZE,
        "available_counts": {
            "benign": benign_available,
            "malicious": malicious_available,
        },
    }
    with open(sweep_root / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep_manifest, f, indent=2)
    with open(mode_dir / "sweep_config.json", "w", encoding="utf-8") as f:
        json.dump(sweep_manifest, f, indent=2)

    for file_path in mode_dir.glob("benign_*_run*.json"):
        file_path.unlink()
    summary_path = mode_dir / "summary.csv"
    if summary_path.exists():
        summary_path.unlink()
    print(f"Existing high-benign sweep outputs for mode {mode_id} will be overwritten.\n")

    summaries = []
    benign_pcts = range(benign_start_pct, benign_end_pct + 1, benign_step_pct)

    for benign_pct in benign_pcts:
        benign_ratio = benign_pct / 100.0
        pending_runs = list(range(1, repeats + 1))

        print(f"\n[Benign Ratio = {benign_ratio:.2f}]")

        for run_idx in pending_runs:
            print(f"  -> Repeat {run_idx}/{repeats}")
            rng = random.Random(cfg.RANDOM_SEED + mode_id * 10000 + benign_pct * 100 + run_idx)
            sampled, sample_meta = _sample_high_benign_subset(
                test_data=test_data,
                total_size=total_size,
                benign_ratio=benign_ratio,
                rng=rng,
            )

            print(
                "     "
                f"fixed_total_size={total_size}, "
                f"benign={sample_meta['benign_count']}, "
                f"malicious={sample_meta['malicious_count']}, "
                f"attack_mix={sample_meta['attack_type_counts']}"
            )

            sequences = [d["sequence"] for d in sampled]
            labels = np.array([d["label"] for d in sampled])
            from models.detectors.pipeline import DetectionPipeline

            pipeline = DetectionPipeline(
                config_path=config_path,
                test_mode=mode_id,
                layer3_variant=layer3_variant,
            )
            if "layer1" in pipeline.layers:
                pipeline.layers["layer1"].threshold = float(threshold)
            results = pipeline.predict(
                sequences=sequences,
                labels=labels,
                return_details=True,
            )

            results["sweep_metadata"] = {
                "sweep_type": "high_benign_distribution",
                "fixed_total_size": total_size,
                "requested_benign_pct": benign_pct,
                "actual_benign_ratio": sample_meta["actual_benign_ratio"],
                "benign_count": sample_meta["benign_count"],
                "malicious_count": sample_meta["malicious_count"],
                "attack_type_counts": sample_meta["attack_type_counts"],
                "repeat_index": run_idx,
                "repeats": repeats,
                "layer3_variant": layer3_variant or "default",
                "selected_layer3_variant": results.get("selected_layer3_variant", "layer3"),
                "layer1_threshold_from_config": threshold,
                "uses_detection_pipeline": True,
            }

            out_path = mode_dir / f"benign_{benign_pct}_run{run_idx}.json"
            save_results(results, out_path)

            layer3_processed = results["layer_stats"].get("layer3", {}).get("processed", 0)
            summaries.append({
                "ratio": benign_ratio,
                "benign_pct": benign_pct,
                "run": run_idx,
                "layer3_variant": layer3_variant or "default",
                "fixed_total_size": total_size,
                "L3_ratio": layer3_processed / len(sequences),
                "time": results["time"],
                "precision": results["metrics"]["precision"],
                "recall": results["metrics"]["recall"],
                "f1": results["metrics"]["f1_score"],
                "fpr": results["metrics"]["fpr"],
                "fnr": results["metrics"]["fnr"],
            })

    all_rows = []
    for file_path in sorted(mode_dir.glob("benign_*_run*.json")):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        meta = data.get("sweep_metadata", {})
        layer3_processed = data.get("layer_stats", {}).get("layer3", {}).get("processed", 0)
        num_samples = int(data.get("num_samples", total_size))
        all_rows.append({
            "ratio": float(meta.get("actual_benign_ratio", meta.get("requested_benign_pct", 0) / 100.0)),
            "benign_pct": int(meta.get("requested_benign_pct", 0)),
            "run": int(meta.get("repeat_index", 1)),
            "layer3_variant": meta.get("layer3_variant", "default"),
            "fixed_total_size": int(meta.get("fixed_total_size", total_size)),
            "L3_ratio": layer3_processed / num_samples if num_samples else 0.0,
            "time": float(data["time"]),
            "precision": float(data["metrics"]["precision"]),
            "recall": float(data["metrics"]["recall"]),
            "f1": float(data["metrics"]["f1_score"]),
            "fpr": float(data["metrics"]["fpr"]),
            "fnr": float(data["metrics"]["fnr"]),
        })

    if all_rows:
        pd.DataFrame(all_rows).sort_values(["benign_pct", "run"]).to_csv(mode_dir / "summary.csv", index=False)

    print("\n✅ DONE: High-benign distribution sweep completed.")

def run_threshold_sweep(mode_id=11, layer3_variant=None):
    print("\n🚀 RUNNING THRESHOLD SWEEP\n")

    test_path = PROCESSED_DIR / "test_processed.jsonl"
    config_path = CONFIG_DIR / "detection_config.yaml"

    test_data = load_jsonl(test_path)
    if not test_data:
        raise ValueError("Empty test-data")
    if layer3_variant is None:
        layer3_variant = ask_layer3_variant_for_mode(mode_id)

    sequences = [d["sequence"] for d in test_data]
    labels = np.array([d["label"] for d in test_data])

    thresholds = np.arange(0.05, 0.1501, 0.005)

    sweep_dir = RESULT_DIR / "threshold_sweep" / f"mode_{mode_id}{layer3_variant_suffix(layer3_variant)}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    from models.detectors.pipeline import DetectionPipeline
    for th in thresholds:
        print(f"\n[Threshold = {th:.3f}]")

        # Rebuild the pipeline for each threshold to avoid cross-run state.
        pipeline = DetectionPipeline(
            config_path=config_path,
            test_mode=mode_id,
            layer3_variant=layer3_variant,
        )

        pipeline.layers["layer1"].threshold = float(th)

        results = pipeline.predict(
            sequences=sequences,
            labels=labels,
            return_details=True,
        )

        out_path = sweep_dir / f"th_{th:.3f}.json"
        save_results(results, out_path)

        layer3_processed = results["layer_stats"].get("layer3", {}).get("processed", 0)
        summary.append({
            "threshold": th,
            "layer3_variant": layer3_variant or "default",
            "L3_ratio": layer3_processed / len(sequences),
            "time": results["time"]
        })

    df = pd.DataFrame(summary)
    df.to_csv(sweep_dir / "summary.csv", index=False)

    print("\n✅ DONE: Threshold sweep completed.")


def _parse_int_list(raw):
    values = []
    for part in str(raw or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise ValueError("Concurrency values must be >= 1")
        values.append(value)
    return sorted(set(values))


def _threshold_values(start, end, step):
    if step <= 0:
        raise ValueError("Threshold step must be > 0")
    if end < start:
        raise ValueError("Threshold end must be >= start")

    values = []
    current = float(start)
    while current <= float(end) + 1e-12:
        values.append(round(current, 6))
        current += float(step)
    return values


def _backend_latency_current_dir(layer3_variant):
    suffix = layer3_variant_suffix(layer3_variant)
    return RESULT_DIR / "backend_latency_sweep" / f"mode_11{suffix}" / "current"


def _load_backend_latency_test_data(test_percent, random_seed):
    test_path = PROCESSED_DIR / "test_processed.jsonl"
    test_data = load_jsonl(test_path)
    if not test_data:
        raise ValueError("Empty test-data")

    if not (1 <= int(test_percent) <= 100):
        raise ValueError("Test percentage must be in [1, 100]")

    if int(test_percent) < 100:
        rng = random.Random(int(random_seed))
        test_data = list(test_data)
        rng.shuffle(test_data)
        subset_size = max(1, int(len(test_data) * (int(test_percent) / 100.0)))
        test_data = test_data[:subset_size]
        print(f"Using shuffled test subset: {subset_size} samples")

    return test_data


def _latency_stats(events, baseline_seconds=None, baseline_by_local_index=None):
    values = np.array([float(event["observed_latency_seconds"]) for event in events], dtype=float)
    if values.size == 0:
        return {
            "count": 0,
            "mean_observed_seconds": 0.0,
            "p50_observed_seconds": 0.0,
            "p95_observed_seconds": 0.0,
            "max_observed_seconds": 0.0,
            "baseline_service_seconds": baseline_seconds,
            "mean_effective_backend_delay_seconds": None,
            "p50_effective_backend_delay_seconds": None,
            "p95_effective_backend_delay_seconds": None,
            "mean_paired_effective_backend_delay_seconds": None,
            "p50_paired_effective_backend_delay_seconds": None,
            "p95_paired_effective_backend_delay_seconds": None,
        }

    stats = {
        "count": int(values.size),
        "mean_observed_seconds": float(values.mean()),
        "p50_observed_seconds": float(np.percentile(values, 50)),
        "p95_observed_seconds": float(np.percentile(values, 95)),
        "max_observed_seconds": float(values.max()),
        "baseline_service_seconds": baseline_seconds,
        "mean_effective_backend_delay_seconds": None,
        "p50_effective_backend_delay_seconds": None,
        "p95_effective_backend_delay_seconds": None,
        "mean_paired_effective_backend_delay_seconds": None,
        "p50_paired_effective_backend_delay_seconds": None,
        "p95_paired_effective_backend_delay_seconds": None,
    }

    if baseline_seconds is not None:
        effective = np.maximum(values - float(baseline_seconds), 0.0)
        stats.update({
            "mean_effective_backend_delay_seconds": float(effective.mean()),
            "p50_effective_backend_delay_seconds": float(np.percentile(effective, 50)),
            "p95_effective_backend_delay_seconds": float(np.percentile(effective, 95)),
        })

    if baseline_by_local_index:
        paired = []
        for event in events:
            local_idx = int(event["local_index"])
            if local_idx not in baseline_by_local_index:
                continue
            paired.append(
                max(
                    0.0,
                    float(event["observed_latency_seconds"])
                    - float(baseline_by_local_index[local_idx]),
                )
            )

        if paired:
            paired_values = np.array(paired, dtype=float)
            stats.update({
                "mean_paired_effective_backend_delay_seconds": float(paired_values.mean()),
                "p50_paired_effective_backend_delay_seconds": float(np.percentile(paired_values, 50)),
                "p95_paired_effective_backend_delay_seconds": float(np.percentile(paired_values, 95)),
            })

    return stats


def _run_mode11_until_layer3(pipeline, sequences, labels):
    n = len(sequences)
    scores = np.zeros(n, dtype=float)
    final_before_l3 = np.zeros(n, dtype=int)
    layer_stats = {}

    stage_start = time.perf_counter()

    t1 = time.perf_counter()
    out1 = pipeline._run_layer("layer1", sequences, batch_size=128)
    layer1_elapsed = time.perf_counter() - t1
    p1 = out1["predictions"]
    s1 = out1["probabilities"]
    scores[:] = s1
    idx_susp = np.where(p1 == 1)[0]

    layer_stats["layer1"] = {
        "processed": n,
        "forwarded": int((p1 == 0).sum()),
        "sent_to_next": int((p1 == 1).sum()),
        "time": float(layer1_elapsed),
    }
    pipeline._attach_layer_confusion(layer_stats, "layer1", labels, p1)

    idx_to_l3 = np.array([], dtype=int)
    layer2_elapsed = 0.0
    if len(idx_susp) > 0:
        sub2 = [sequences[i] for i in idx_susp]
        t2 = time.perf_counter()
        out2 = pipeline._run_layer("layer2", sub2, batch_size=4)
        layer2_elapsed = time.perf_counter() - t2
        p2 = out2["predictions"]
        s2 = out2["probabilities"]
        scores[idx_susp] = s2

        idx_block = idx_susp[p2 == 1]
        final_before_l3[idx_block] = 1
        idx_to_l3 = idx_susp[p2 == 0]

        layer_stats["layer2"] = {
            "processed": int(len(idx_susp)),
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
            "time": float(layer2_elapsed),
        }
        labels2 = labels[idx_susp] if labels is not None else None
        pipeline._attach_layer_confusion(layer_stats, "layer2", labels2, p2)
    else:
        layer_stats["layer2"] = {
            "processed": 0,
            "blocked": 0,
            "sent_to_next": 0,
            "time": 0.0,
        }

    return {
        "scores": scores,
        "final_before_l3": final_before_l3,
        "idx_to_l3": idx_to_l3,
        "layer_stats": layer_stats,
        "l1_l2_wall_time_seconds": float(time.perf_counter() - stage_start),
        "layer1_time_seconds": float(layer1_elapsed),
        "layer2_time_seconds": float(layer2_elapsed),
        "layer1_embedding_cache_loaded": bool(out1["raw"].get("used_cached_embeddings", False)),
        "layer1_embedding_batch_size": int(out1["raw"].get("embedding_batch_size", 128)),
    }


def _call_layer3_for_latency(layer3, sequence):
    if hasattr(layer3, "_call_llm"):
        result = layer3._call_llm(str(sequence or "").strip())
        confidence = float(result.get("confidence", 0.0) or 0.0)
        is_attack = bool(result.get("is_malicious", False)) and confidence >= float(getattr(layer3, "threshold", 0.5))
        return int(is_attack), confidence, result

    detected, score = layer3.detect_single(str(sequence or "").strip())
    return int(bool(detected)), float(score), {"reason": "detect_single"}


def _replay_layer3_backend(
    layer3,
    sequences,
    labels,
    sample_indices,
    agent_concurrency,
):
    events = []
    predictions_by_agent = [
        np.zeros(len(sequences), dtype=int)
        for _ in range(int(agent_concurrency))
    ]
    probabilities_by_agent = [
        np.zeros(len(sequences), dtype=float)
        for _ in range(int(agent_concurrency))
    ]
    details_by_agent = [
        [None] * len(sequences)
        for _ in range(int(agent_concurrency))
    ]
    agent_streams = []
    wall_start = time.perf_counter()

    def agent_stream(agent_id):
        stream_events = []
        stream_start = time.perf_counter()
        for local_idx, sequence in enumerate(sequences):
            label = labels[local_idx] if labels is not None else None
            t_submit_wall = time.time()
            t_submit_perf = time.perf_counter()
            try:
                pred, score, detail = _call_layer3_for_latency(layer3, sequence)
                error = None
            except Exception as exc:
                pred = 0
                score = 0.0
                detail = {"reason": f"Layer3 replay error: {str(exc)[:120]}"}
                error = str(exc)[:240]
            t_receive_perf = time.perf_counter()

            stream_events.append({
                "agent_id": int(agent_id),
                "agent_concurrency": int(agent_concurrency),
                "local_index": int(local_idx),
                "sample_index": int(sample_indices[local_idx]),
                "label": int(label) if label is not None else None,
                "t_send_epoch": float(t_submit_wall),
                "t_receive_epoch": float(t_submit_wall + (t_receive_perf - t_submit_perf)),
                "observed_latency_seconds": float(t_receive_perf - t_submit_perf),
                "prediction": int(pred),
                "probability": float(score),
                "error": error,
                "detail": detail,
            })

        stream_elapsed = time.perf_counter() - stream_start
        stream_values = np.array(
            [float(event["observed_latency_seconds"]) for event in stream_events],
            dtype=float,
        )
        stream_summary = {
            "agent_id": int(agent_id),
            "call_count": int(len(stream_events)),
            "wall_time_seconds": float(stream_elapsed),
            "mean_observed_seconds": float(stream_values.mean()) if stream_values.size else 0.0,
            "p50_observed_seconds": float(np.percentile(stream_values, 50)) if stream_values.size else 0.0,
            "p95_observed_seconds": float(np.percentile(stream_values, 95)) if stream_values.size else 0.0,
        }
        return int(agent_id), stream_events, stream_summary

    with ThreadPoolExecutor(max_workers=int(agent_concurrency)) as executor:
        futures = [
            executor.submit(agent_stream, agent_id)
            for agent_id in range(int(agent_concurrency))
        ]

        for future in as_completed(futures):
            agent_id, stream_events, stream_summary = future.result()
            agent_streams.append(stream_summary)
            for event in stream_events:
                local_idx = event["local_index"]
                predictions_by_agent[agent_id][local_idx] = event["prediction"]
                probabilities_by_agent[agent_id][local_idx] = event["probability"]
                details_by_agent[agent_id][local_idx] = event["detail"]
            events.extend(stream_events)

    events.sort(key=lambda row: (row["agent_id"], row["local_index"]))
    agent_streams.sort(key=lambda row: row["agent_id"])
    return {
        "predictions": predictions_by_agent[0] if predictions_by_agent else np.array([], dtype=int),
        "probabilities": probabilities_by_agent[0] if probabilities_by_agent else np.array([], dtype=float),
        "details": details_by_agent[0] if details_by_agent else [],
        "predictions_by_agent": predictions_by_agent,
        "probabilities_by_agent": probabilities_by_agent,
        "agent_streams": agent_streams,
        "events": events,
        "wall_time_seconds": float(time.perf_counter() - wall_start),
    }


def _gd2_result_filename(threshold_tag, agent_concurrency):
    return f"th_{threshold_tag}_agentc{agent_concurrency}.json"


def _gd2_events_filename(threshold_tag, agent_concurrency):
    return f"l3_events_th_{threshold_tag}_agentc{agent_concurrency}.jsonl"


def _load_existing_gd2_baseline(gd2_dir, threshold_tag):
    result_path = gd2_dir / _gd2_result_filename(threshold_tag, 1)
    events_path = gd2_dir / _gd2_events_filename(threshold_tag, 1)
    if not (result_path.exists() and events_path.exists()):
        return None, None

    result = _read_json(result_path)
    events = load_jsonl(events_path)
    baseline_seconds = (
        result.get("l3_backend_latency", {}).get("p50_observed_seconds")
        or result.get("sweep_metadata", {}).get("baseline_service_seconds")
    )
    if baseline_seconds is None:
        baseline_seconds = _latency_stats(events)["p50_observed_seconds"]

    baseline_by_local_index = {
        int(event["local_index"]): float(event["observed_latency_seconds"])
        for event in events
        if int(event.get("agent_id", 0)) == 0
    }
    return float(baseline_seconds), baseline_by_local_index


def _summary_row_from_gd2_result(result, threshold, agent_concurrency, result_file):
    metrics = result.get("metrics", {}) or {}
    metadata = result.get("sweep_metadata", {}) or {}
    timing = result.get("timing_components", {}) or {}
    latency = result.get("l3_backend_latency", {}) or {}
    num_samples = int(result.get("num_samples", metadata.get("num_samples", 0)) or 0)
    pipeline_total = float(
        timing.get("pipeline_total_time_seconds", result.get("raw_observed_wall_time", result.get("time", 0.0))) or 0.0
    )

    return {
        "threshold": float(threshold),
        "backend_concurrency": int(agent_concurrency),
        "agent_concurrency": int(agent_concurrency),
        "layer3_variant": metadata.get("layer3_variant", result.get("selected_layer3_variant", "default")),
        "num_samples": num_samples,
        "l2_processed": int(metadata.get("l2_processed", 0) or 0),
        "l2_ratio": float(metadata.get("l2_ratio", 0.0) or 0.0),
        "l3_processed": int(metadata.get("l3_processed", metadata.get("l3_candidates", 0)) or 0),
        "l3_calls_total": int(metadata.get("l3_calls_total", 0) or 0),
        "l3_ratio": float(metadata.get("l3_ratio", 0.0) or 0.0),
        "l1_l2_wall_time_seconds": float(timing.get("l1_l2_wall_time_seconds", 0.0) or 0.0),
        "l3_backend_wall_time_seconds": float(
            timing.get("l3_backend_wall_time_seconds", latency.get("observed_backend_wall_time_seconds", 0.0)) or 0.0
        ),
        "pipeline_total_time_seconds": pipeline_total,
        "pipeline_latency_per_sample_seconds": float(
            timing.get(
                "pipeline_latency_per_sample_seconds",
                pipeline_total / num_samples if num_samples else 0.0,
            )
            or 0.0
        ),
        "precision": float(metrics.get("precision", 0.0) or 0.0),
        "recall": float(metrics.get("recall", 0.0) or 0.0),
        "f1": float(metrics.get("f1_score", metrics.get("f1", 0.0)) or 0.0),
        "fpr": float(metrics.get("fpr", 0.0) or 0.0),
        "fnr": float(metrics.get("fnr", 0.0) or 0.0),
        "baseline_source_concurrency": int(metadata.get("baseline_source_concurrency", 1) or 1),
        "result_file": result_file,
        "count": int(latency.get("count", 0) or 0),
        "mean_observed_seconds": float(latency.get("mean_observed_seconds", 0.0) or 0.0),
        "p50_observed_seconds": float(latency.get("p50_observed_seconds", 0.0) or 0.0),
        "p95_observed_seconds": float(latency.get("p95_observed_seconds", 0.0) or 0.0),
        "max_observed_seconds": float(latency.get("max_observed_seconds", 0.0) or 0.0),
        "baseline_service_seconds": latency.get("baseline_service_seconds"),
        "mean_effective_backend_delay_seconds": latency.get("mean_effective_backend_delay_seconds"),
        "p50_effective_backend_delay_seconds": latency.get("p50_effective_backend_delay_seconds"),
        "p95_effective_backend_delay_seconds": latency.get("p95_effective_backend_delay_seconds"),
        "mean_paired_effective_backend_delay_seconds": latency.get("mean_paired_effective_backend_delay_seconds"),
        "p50_paired_effective_backend_delay_seconds": latency.get("p50_paired_effective_backend_delay_seconds"),
        "p95_paired_effective_backend_delay_seconds": latency.get("p95_paired_effective_backend_delay_seconds"),
    }


def _write_gd2_summary(summary_rows, gd2_dir, current_dir):
    if not summary_rows:
        return
    summary = pd.DataFrame(summary_rows).sort_values(["threshold", "agent_concurrency"])
    summary.to_csv(gd2_dir / "summary.csv", index=False)
    summary.to_csv(current_dir / "summary.csv", index=False)


def _run_backend_latency_gd1(layer3_variant, thresholds, test_percent, random_seed, overwrite=False):
    print("\n[GD_1] Running Mode 11 up to Layer 1 + Layer 2 only.")

    layer3_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _backend_latency_current_dir(layer3_variant)
    if current_dir.exists():
        if not overwrite:
            print("[GD_1] Existing current/ artifacts found; keeping them unchanged.")
            print(f"Output: {current_dir / 'gd1'}")
            return current_dir
        print("[GD_1] Overwriting existing current/ artifacts for this variant.\n")
        shutil.rmtree(current_dir)

    gd1_dir = current_dir / "gd1"
    gd1_dir.mkdir(parents=True, exist_ok=True)

    test_data = _load_backend_latency_test_data(test_percent, random_seed)
    sequences = [d["sequence"] for d in test_data]
    labels = np.array([d["label"] for d in test_data], dtype=int)
    config_path = CONFIG_DIR / "detection_config.yaml"

    manifest = {
        "stage": "GD_1",
        "sweep_type": "layer3_backend_latency_staged",
        "mode_id": 11,
        "mode": "layer1_to_layer2_to_layer3",
        "layer3_variant": layer3_variant or "default",
        "thresholds": list(thresholds),
        "test_percent": int(test_percent),
        "num_samples": int(len(sequences)),
        "random_seed": int(random_seed),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "artifact_contract": (
            "GD_1 freezes L1/L2 routing and timing. GD_2 can combine these "
            "artifacts with L3 replay timing without rerunning L1/L2."
        ),
    }
    _write_json(gd1_dir / "config.json", manifest)
    _write_json(current_dir / "sweep_config.json", manifest)

    from models.detectors.pipeline import DetectionPipeline

    summary_rows = []
    for threshold in thresholds:
        threshold_tag = f"{float(threshold):.3f}"
        print(f"\n[GD_1 threshold = {threshold_tag}]")

        pipeline = DetectionPipeline(
            config_path=config_path,
            test_mode=11,
            layer3_variant=layer3_variant,
        )
        pipeline.layers["layer1"].threshold = float(threshold)

        partial = _run_mode11_until_layer3(pipeline, sequences, labels)
        idx_to_l3 = np.array(partial["idx_to_l3"], dtype=int)
        layer1_stats = partial["layer_stats"].get("layer1", {})
        layer2_stats = partial["layer_stats"].get("layer2", {})

        candidates_file = f"l3_candidates_th_{threshold_tag}.jsonl"
        candidate_rows = []
        for local_idx, sample_idx in enumerate(idx_to_l3):
            source = test_data[int(sample_idx)]
            candidate_rows.append({
                "local_index": int(local_idx),
                "sample_index": int(sample_idx),
                "label": int(labels[int(sample_idx)]),
                "attack_type": str(source.get("attack_type", "")),
                "sequence": source["sequence"],
            })
        _write_jsonl(gd1_dir / candidates_file, candidate_rows)

        l2_processed = int(layer1_stats.get("sent_to_next", 0))
        l3_processed = int(len(idx_to_l3))
        l2_ratio = float(l2_processed / len(sequences)) if sequences else 0.0
        l3_ratio = float(l3_processed / len(sequences)) if sequences else 0.0
        l1_l2_time = float(partial["l1_l2_wall_time_seconds"])

        threshold_artifact = {
            "stage": "GD_1",
            "threshold": float(threshold),
            "threshold_tag": threshold_tag,
            "num_samples": int(len(sequences)),
            "labels": labels.tolist(),
            "final_before_l3": np.array(partial["final_before_l3"], dtype=int).tolist(),
            "scores_before_l3": np.array(partial["scores"], dtype=float).tolist(),
            "idx_to_l3": idx_to_l3.tolist(),
            "layer_stats": partial["layer_stats"],
            "layer1_embedding_cache_loaded": partial["layer1_embedding_cache_loaded"],
            "layer1_embedding_batch_size": partial["layer1_embedding_batch_size"],
            "layer_implementations": dict(pipeline.layer_aliases),
            "selected_layer3_variant": pipeline.layer_aliases.get("layer3", "layer3"),
            "candidates_file": candidates_file,
            "timing_components": {
                "l1_l2_wall_time_seconds": l1_l2_time,
                "layer1_time_seconds": float(partial["layer1_time_seconds"]),
                "layer2_time_seconds": float(partial["layer2_time_seconds"]),
                "l3_backend_wall_time_seconds": None,
                "pipeline_total_time_seconds": None,
                "pipeline_latency_per_sample_seconds": None,
            },
            "routing": {
                "l2_processed": l2_processed,
                "l2_ratio": l2_ratio,
                "l3_processed": l3_processed,
                "l3_ratio": l3_ratio,
                "blocked_before_l3": int(np.array(partial["final_before_l3"], dtype=int).sum()),
            },
        }
        _write_json(gd1_dir / f"threshold_{threshold_tag}.json", threshold_artifact)

        summary_rows.append({
            "threshold": float(threshold),
            "layer3_variant": layer3_variant or "default",
            "num_samples": int(len(sequences)),
            "l2_processed": l2_processed,
            "l2_ratio": l2_ratio,
            "l3_processed": l3_processed,
            "l3_ratio": l3_ratio,
            "blocked_before_l3": int(np.array(partial["final_before_l3"], dtype=int).sum()),
            "l1_l2_wall_time_seconds": l1_l2_time,
            "layer1_time_seconds": float(partial["layer1_time_seconds"]),
            "layer2_time_seconds": float(partial["layer2_time_seconds"]),
            "candidates_file": candidates_file,
            "threshold_artifact": f"threshold_{threshold_tag}.json",
        })

    summary = pd.DataFrame(summary_rows).sort_values("threshold")
    summary.to_csv(gd1_dir / "summary.csv", index=False)
    summary.to_csv(current_dir / "gd1_summary.csv", index=False)

    print(f"\n✅ GD_1 completed.")
    print(f"Output: {gd1_dir}")
    return current_dir


def _run_backend_latency_gd2(layer3_variant, target_agent_concurrency, overwrite=False):
    print("\n[GD_2] Replaying Layer 3 from saved GD_1 artifacts.")
    if overwrite:
        print("[GD_2] Existing GD_2 artifacts will be overwritten.\n")
    else:
        print("[GD_2] Existing completed artifacts will be reused and skipped.\n")

    layer3_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _backend_latency_current_dir(layer3_variant)
    gd1_dir = current_dir / "gd1"
    gd2_dir = current_dir / "gd2"

    if not (gd1_dir / "config.json").exists():
        raise FileNotFoundError(
            f"GD_1 artifacts not found: {gd1_dir}. Run GD_1 first for this Layer 3 variant."
        )
    if overwrite and gd2_dir.exists():
        shutil.rmtree(gd2_dir)
    if overwrite and (current_dir / "summary.csv").exists():
        (current_dir / "summary.csv").unlink()
    gd2_dir.mkdir(parents=True, exist_ok=True)

    if int(target_agent_concurrency) < 1:
        raise ValueError("Target Layer 3 agent concurrency must be >= 1")

    gd1_config = _read_json(gd1_dir / "config.json")
    layer3_variant = _normalize_layer3_variant(gd1_config.get("layer3_variant"))
    thresholds = [float(x) for x in gd1_config.get("thresholds", [])]
    replay_concurrency_values = [1] if int(target_agent_concurrency) == 1 else [1, int(target_agent_concurrency)]

    gd2_config = {
        "stage": "GD_2",
        "sweep_type": "layer3_backend_latency_staged",
        "mode_id": 11,
        "mode": "layer1_to_layer2_to_layer3",
        "layer3_variant": layer3_variant or "default",
        "baseline_agent_concurrency": 1,
        "target_agent_concurrency": int(target_agent_concurrency),
        "concurrency_values": replay_concurrency_values,
        "gd1_config": "gd1/config.json",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "measurement_note": (
            "GD_2 reuses frozen GD_1 L1/L2 routing. pipeline_total_time_seconds "
            "= GD_1 l1_l2_wall_time_seconds + GD_2 l3_backend_wall_time_seconds."
        ),
    }
    _write_json(gd2_dir / "config.json", gd2_config)

    from models.detectors.pipeline import DetectionPipeline
    from utils.metrics import (
        compute_complete_metrics,
        compute_roc_curve_data,
        compute_precision_recall_curve_data,
    )

    pipeline = DetectionPipeline(
        config_path=CONFIG_DIR / "detection_config.yaml",
        test_mode=11,
        layer3_variant=layer3_variant,
    )

    summary_rows = []
    for threshold in thresholds:
        threshold_tag = f"{float(threshold):.3f}"
        print(f"\n[GD_2 threshold = {threshold_tag}]")

        gd1_threshold = _read_json(gd1_dir / f"threshold_{threshold_tag}.json")
        candidate_rows = load_jsonl(gd1_dir / gd1_threshold["candidates_file"])

        labels = np.array(gd1_threshold["labels"], dtype=int)
        final_before_l3 = np.array(gd1_threshold["final_before_l3"], dtype=int)
        scores_before_l3 = np.array(gd1_threshold["scores_before_l3"], dtype=float)
        idx_to_l3 = np.array(gd1_threshold["idx_to_l3"], dtype=int)
        l3_sequences = [row["sequence"] for row in candidate_rows]
        l3_labels = np.array([row["label"] for row in candidate_rows], dtype=int)
        sample_indices = np.array([row["sample_index"] for row in candidate_rows], dtype=int)

        baseline_seconds = None
        baseline_by_local_index = None

        for agent_concurrency in replay_concurrency_values:
            out_file = _gd2_result_filename(threshold_tag, agent_concurrency)
            events_file = _gd2_events_filename(threshold_tag, agent_concurrency)
            out_path = gd2_dir / out_file
            events_path = gd2_dir / events_file

            print(
                f"  -> agent concurrency={agent_concurrency}, "
                f"L3 candidates per agent={len(l3_sequences)}"
            )

            if out_path.exists() and events_path.exists():
                print(f"     skip existing: {out_file}")
                existing_result = _read_json(out_path)
                if int(agent_concurrency) == 1 and baseline_seconds is None:
                    baseline_seconds, baseline_by_local_index = _load_existing_gd2_baseline(
                        gd2_dir,
                        threshold_tag,
                    )
                summary_rows.append(
                    _summary_row_from_gd2_result(
                        existing_result,
                        threshold,
                        agent_concurrency,
                        out_file,
                    )
                )
                _write_gd2_summary(summary_rows, gd2_dir, current_dir)
                continue

            if out_path.exists() or events_path.exists():
                print(f"     incomplete existing artifact found; rerunning: {out_file}")

            replay = _replay_layer3_backend(
                pipeline.layers["layer3"],
                l3_sequences,
                l3_labels,
                sample_indices,
                agent_concurrency,
            )

            if baseline_seconds is None:
                first_stats = _latency_stats(replay["events"])
                baseline_seconds = first_stats["p50_observed_seconds"]
                baseline_by_local_index = {
                    int(event["local_index"]): float(event["observed_latency_seconds"])
                    for event in replay["events"]
                    if int(event.get("agent_id", 0)) == 0
                }

            latency = _latency_stats(
                replay["events"],
                baseline_seconds=baseline_seconds,
                baseline_by_local_index=baseline_by_local_index,
            )

            final = np.array(final_before_l3, dtype=int)
            scores = np.array(scores_before_l3, dtype=float)
            p3 = replay["predictions"] if len(idx_to_l3) else np.array([], dtype=int)
            s3 = replay["probabilities"] if len(idx_to_l3) else np.array([], dtype=float)
            if len(idx_to_l3):
                final[idx_to_l3[p3 == 1]] = 1
                scores[idx_to_l3] = s3

            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

            layer_stats = dict(gd1_threshold["layer_stats"])
            layer_stats["layer3"] = {
                "processed": int(len(idx_to_l3)),
                "blocked": int(p3.sum()) if len(idx_to_l3) else 0,
                "forwarded": int((p3 == 0).sum()) if len(idx_to_l3) else 0,
                "time": float(replay["wall_time_seconds"]),
                "backend_concurrency": int(agent_concurrency),
                "agent_concurrency": int(agent_concurrency),
                "agent_streams": replay["agent_streams"],
                "latency": latency,
            }
            pipeline._attach_layer_confusion(layer_stats, "layer3", l3_labels, p3)

            l1_l2_time = float(gd1_threshold["timing_components"]["l1_l2_wall_time_seconds"])
            l3_time = float(replay["wall_time_seconds"])
            total_time = float(l1_l2_time + l3_time)
            num_samples = int(gd1_threshold["num_samples"])

            result = {
                "test_mode": "layer1_to_layer2_to_layer3_backend_latency_sweep",
                "test_mode_id": 11,
                "strategy": "staged_chained_backend_replay",
                "predictions": final,
                "probabilities": scores,
                "num_samples": num_samples,
                "num_detected": int(final.sum()),
                "time": total_time,
                "raw_observed_wall_time": total_time,
                "layer_stats": layer_stats,
                "metrics": metrics,
                "layer1_embedding_cache_loaded": gd1_threshold["layer1_embedding_cache_loaded"],
                "layer1_embedding_batch_size": gd1_threshold["layer1_embedding_batch_size"],
                "layer_implementations": dict(gd1_threshold["layer_implementations"]),
                "selected_layer3_variant": gd1_threshold["selected_layer3_variant"],
                "sweep_metadata": {
                    **gd2_config,
                    "threshold": float(threshold),
                    "backend_concurrency": int(agent_concurrency),
                    "agent_concurrency": int(agent_concurrency),
                    "l2_processed": int(gd1_threshold["routing"]["l2_processed"]),
                    "l2_ratio": float(gd1_threshold["routing"]["l2_ratio"]),
                    "l3_candidates": int(len(idx_to_l3)),
                    "l3_processed": int(len(idx_to_l3)),
                    "l3_calls_total": int(len(idx_to_l3) * agent_concurrency),
                    "l3_ratio": float(gd1_threshold["routing"]["l3_ratio"]),
                    "baseline_source_concurrency": 1,
                    "baseline_service_seconds": float(baseline_seconds),
                    "gd1_threshold_artifact": f"gd1/threshold_{threshold_tag}.json",
                    "events_file": events_file,
                    "candidates_file": f"gd1/{gd1_threshold['candidates_file']}",
                    "uses_detection_pipeline": True,
                    "save_results_format": True,
                },
                "timing_components": {
                    "l1_l2_wall_time_seconds": l1_l2_time,
                    "layer1_time_seconds": float(gd1_threshold["timing_components"]["layer1_time_seconds"]),
                    "layer2_time_seconds": float(gd1_threshold["timing_components"]["layer2_time_seconds"]),
                    "l3_backend_wall_time_seconds": l3_time,
                    "pipeline_total_time_seconds": total_time,
                    "pipeline_latency_per_sample_seconds": float(total_time / num_samples) if num_samples else 0.0,
                },
                "l3_backend_latency": {
                    **latency,
                    "observed_backend_wall_time_seconds": l3_time,
                    "baseline_source_concurrency": 1,
                    "agent_concurrency": int(agent_concurrency),
                    "agent_streams": replay["agent_streams"],
                },
            }

            save_results(result, out_path)
            _write_jsonl(events_path, replay["events"])

            summary_rows.append({
                "threshold": float(threshold),
                "backend_concurrency": int(agent_concurrency),
                "agent_concurrency": int(agent_concurrency),
                "layer3_variant": layer3_variant or "default",
                "num_samples": num_samples,
                "l2_processed": int(gd1_threshold["routing"]["l2_processed"]),
                "l2_ratio": float(gd1_threshold["routing"]["l2_ratio"]),
                "l3_processed": int(len(idx_to_l3)),
                "l3_calls_total": int(len(idx_to_l3) * agent_concurrency),
                "l3_ratio": float(gd1_threshold["routing"]["l3_ratio"]),
                "l1_l2_wall_time_seconds": l1_l2_time,
                "l3_backend_wall_time_seconds": l3_time,
                "pipeline_total_time_seconds": total_time,
                "pipeline_latency_per_sample_seconds": float(total_time / num_samples) if num_samples else 0.0,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1_score"]),
                "fpr": float(metrics["fpr"]),
                "fnr": float(metrics["fnr"]),
                "baseline_source_concurrency": 1,
                "result_file": out_file,
                **latency,
            })
            _write_gd2_summary(summary_rows, gd2_dir, current_dir)

    _write_gd2_summary(summary_rows, gd2_dir, current_dir)

    print(f"\n✅ GD_2 completed.")
    print(f"Output: {gd2_dir}")
    print(f"Combined summary: {current_dir / 'summary.csv'}")
    return current_dir


def run_layer3_backend_latency_sweep():
    print("\n🚀 RUNNING LAYER 3 BACKEND LATENCY SWEEP\n")
    print("Staged Issue 2 flow:")
    print("  GD_1 freezes Mode 11 Layer 1/2 routing and timing.")
    print("  GD_2 replays only Layer 3 with con=1 baseline and con=n contention.\n")

    stage_choice = ask_choice(
        "LAYER 3 BACKEND LATENCY SWEEP STAGES",
        [
            ("1", "Run GD_1 only: L1/L2 threshold sweep, no Layer 3 calls"),
            ("2", "Run GD_2 only: Layer 3 replay from saved GD_1"),
            ("3", "Run GD_1 + GD_2"),
            ("0", "Back"),
        ],
    )
    if stage_choice == "0":
        return

    layer3_variant = ask_layer3_variant_for_mode(11)
    normalized_variant = _normalize_layer3_variant(layer3_variant)
    current_dir = _backend_latency_current_dir(normalized_variant)

    if stage_choice in {"1", "3"}:
        gd1_exists = (current_dir / "gd1" / "config.json").exists()
        overwrite_gd1 = False
        if gd1_exists:
            overwrite_gd1 = ask_yes_no(
                f"Existing GD_1 threshold workload found at {current_dir / 'gd1'}. Overwrite it?",
                False,
            )
        if gd1_exists and not overwrite_gd1:
            print(f"Keeping existing GD_1 workload: {current_dir / 'gd1'}")
        else:
            threshold_start = float(ask_with_default("Layer 1 threshold start [default: 0.02]: ", "0.02"))
            threshold_end = float(ask_with_default("Layer 1 threshold end [default: 0.10]: ", "0.10"))
            threshold_step = float(ask_with_default("Layer 1 threshold step [default: 0.02]: ", "0.02"))
            test_percent = int(ask_with_default("Test percentage [default: 100]: ", "100"))
            random_seed = int(ask_with_default("Random seed for test subset [default: 42]: ", "42"))
            thresholds = _threshold_values(threshold_start, threshold_end, threshold_step)
            _run_backend_latency_gd1(
                layer3_variant,
                thresholds,
                test_percent,
                random_seed,
                overwrite=overwrite_gd1,
            )

    if stage_choice in {"2", "3"}:
        gd2_exists = (current_dir / "gd2").exists() and any((current_dir / "gd2").glob("*.json"))
        overwrite_gd2 = False
        if gd2_exists:
            overwrite_gd2 = ask_yes_no(
                f"Existing GD_2 threshold backend results found at {current_dir / 'gd2'}. Overwrite them?",
                False,
            )
        target_agent_concurrency = int(
            ask_with_default("Target Layer 3 agent concurrency n [default: 4]: ", "4")
        )
        _run_backend_latency_gd2(
            layer3_variant,
            target_agent_concurrency,
            overwrite=overwrite_gd2,
        )


def find_threshold_layer1():
    """
    Automatically search Layer1 anomaly threshold that achieves FN = 0
    while minimizing FP.
    """

    print("\n=== FINDING OPTIMAL THRESHOLD FOR LAYER 1 ===\n")

    test_path = PROCESSED_DIR / "test_processed.jsonl"
    config_path = CONFIG_DIR / "detection_config.yaml"

    test_data = load_jsonl(test_path)

    if not test_data:
        raise ValueError("Empty test-data JSONL")

    sequences = [d["sequence"] for d in test_data]
    labels = np.array([d["label"] for d in test_data])

    from models.detectors.pipeline import DetectionPipeline

    pipeline = DetectionPipeline(config_path=config_path, test_mode=1)

    result = pipeline.find_threshold_no_fn(
        sequences=sequences,
        labels=labels
    )

    print("\nOptimal Threshold Found:")
    print("--------------------------------")
    print(f"Threshold : {result['threshold']}")
    print(f"FP Count  : {result['fp']}")
    print("--------------------------------\n")

    save_results(result, RESULT_DIR / "layer1_threshold_search.json")

    print("Threshold search results saved.\n")

def run_without_defense_prompt_injection_eval():
    prompt_injection_eval()

def run_without_defense_other_eval():
    filter_attack_type()

def run_within_defense_eval():
    eval_with_defense()
    eval_with_baseline()
    

def run_figure_generation():
    generate_full_figures()

def run_fine_tune_modermBERT():
    from config.config import ModernBERTConfig
    print_header("LAYER 1 ENCODER FINE-TUNING")
    print("Choose how much of train_processed.jsonl to use for ModernBERT encoder training.\n")
    print(f"CUDA devices from config: {ModernBERTConfig.CUDA_VISIBLE_DEVICES}\n")
    percent = ask_percentage()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(ModernBERTConfig.CUDA_VISIBLE_DEVICES)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.modernbert_finetune",
            "--train_percent",
            str(percent),
        ],
        check=True,
        env=env,
    )

def run_training_autoencoder():
    from config.config import AutoEncoderConfig
    print_header("LAYER 1 AUTOENCODER TRAINING")
    print(f"CUDA devices from config: {AutoEncoderConfig.CUDA_VISIBLE_DEVICES}\n")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(AutoEncoderConfig.CUDA_VISIBLE_DEVICES)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.autoencoder_training",
        ],
        check=True,
        env=env,
    )

def run_finetune_protectai():
    from config.config import Layer2ProtectAIConfig
    print_header("LAYER 2 PROTECTAI FINE-TUNING")
    print("Choose how much of train_processed.jsonl to use for Layer 2 training.")
    print("Correlated attacks will be excluded automatically from both train and validation.\n")
    print(f"CUDA devices from config: {Layer2ProtectAIConfig.CUDA_VISIBLE_DEVICES}\n")
    percent = ask_percentage()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(Layer2ProtectAIConfig.CUDA_VISIBLE_DEVICES)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "training.protectai_deberta_finetune",
            "--train_percent",
            str(percent),
        ],
        check=True,
        env=env,
    )


def run_finetune_layer3_llama31():
    from config.config import Layer3Llama31SFTConfig
    os.environ["CUDA_VISIBLE_DEVICES"] = str(Layer3Llama31SFTConfig.CUDA_VISIBLE_DEVICES)
    from training.llama31_ollama_finetune import run as llama31_run

    print_header("LAYER 3 LOCAL LLAMA 3.1 FINE-TUNING")
    print("This pipeline instruction-tunes a Layer 3 semantic auditor from processed traces.")
    print("It reads data/processed/train_processed.jsonl and data/processed/val_processed.jsonl.")
    print("You choose how much of the training set to use, from 1% to 100%.")
    print(f"Default Hugging Face base model: {Layer3Llama31SFTConfig.MODEL_NAME}")
    print(f"CUDA devices from config: {Layer3Llama31SFTConfig.CUDA_VISIBLE_DEVICES}")
    print("The exported model and Ollama Modelfile will be saved under models/layer3/.")
    print("The training path uses QLoRA 4-bit to reduce VRAM, then merges the adapter")
    print("back into a normal export so you can deploy it to Ollama later.\n")

    percent = ask_percentage()
    summary = llama31_run(train_percent=percent)

    print("\nRun summary")
    print(f"Train percent : {summary.train_percent}%")
    print(f"Train samples : {summary.train_samples}")
    print(f"Val samples   : {summary.val_samples}")
    print(f"Tuning mode   : {summary.tuning_mode}")
    print(f"Output dir    : {summary.output_dir}")
    print(f"HF export     : {summary.hf_export_dir}")
    print(f"Modelfile     : {summary.ollama_modelfile}")


def _list_layer3_finetune_runs():
    root = Path("models/layer3")
    if not root.exists():
        return []

    runs = []
    for run_dir in sorted(root.glob("llama31_8b_layer3_sft_pct*")):
        modelfile = run_dir / "Modelfile"
        hf_dir = run_dir / "hf_model"
        if modelfile.exists() and hf_dir.exists():
            runs.append(run_dir)
    return runs


def select_ollama_quantization():
    default_quant = "q4_K_M"

    print("\nQuantization")
    print(f"Press Enter to use the default: {default_quant}")
    print("Type `c` to choose another quantization option.")
    raw = input("Quantization choice [Enter/c]: ").strip().lower()

    if raw == "":
        return default_quant
    if raw != "c":
        print(f"Unknown choice. Falling back to {default_quant}.")
        return default_quant

    choice = ask_choice(
        "SELECT QUANTIZATION",
        [
            ("1", "q4_K_M (Recommended default)"),
            ("2", "q8_0"),
            ("3", "f16 (no quantization, largest and slowest)"),
            ("4", "q4_0"),
            ("5", "q5_K_M"),
            ("0", "Back to default q4_K_M"),
        ],
    )

    mapping = {
        "1": "q4_K_M",
        "2": "q8_0",
        "3": "f16",
        "4": "q4_0",
        "5": "q5_K_M",
        "0": default_quant,
    }
    return mapping[choice]


def run_deploy_layer3_llama31_to_ollama():
    runs = _list_layer3_finetune_runs()

    print_header("DEPLOY FINE-TUNED LAYER 3 MODEL TO OLLAMA")
    print("This registers one of your fine-tuned Layer 3 runs as a local Ollama model.")
    print("After deployment, you can point detection_config.yaml to the new model name.\n")

    if not runs:
        print("No fine-tuned Layer 3 runs were found under models/layer3/.")
        print("Run 5 -> 4 first to create a train output.\n")
        return

    options = []
    for idx, run_dir in enumerate(runs, start=1):
        training_config_path = run_dir / "training_config.json"
        summary = ""
        if training_config_path.exists():
            try:
                with open(training_config_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                summary = f" ({payload.get('train_percent', '?')}%, {payload.get('tuning_mode', 'unknown')})"
            except Exception:
                summary = ""
        options.append((str(idx), f"{run_dir.name}{summary}"))

    options.append(("0", "Back"))
    choice = ask_choice("SELECT LAYER 3 RUN", options)
    if choice == "0":
        return

    selected_run = runs[int(choice) - 1]
    modelfile_path = selected_run / "Modelfile"
    default_model_name = f"agent-defense-{selected_run.name}"

    print(f"\nSelected run : {selected_run}")
    print(f"Modelfile    : {modelfile_path}")
    quantization = select_ollama_quantization()

    default_model_name_quantized = (
        default_model_name
        if quantization == "f16"
        else f"{default_model_name}-{quantization.lower().replace('_', '-')}"
    )
    model_name = ask_with_default(
        f"Enter Ollama model name [{default_model_name_quantized}]: ",
        default_model_name_quantized,
    )

    print("\nRunning:")
    print(f"ollama create {model_name} -f {modelfile_path} --quantize {quantization}")

    try:
        subprocess.run(
            ["ollama", "create", model_name, "-f", str(modelfile_path), "--quantize", quantization],
            check=True,
        )
    except FileNotFoundError:
        print("\nOllama CLI was not found in PATH.")
        print("Install Ollama or open a shell where `ollama` is available.\n")
        return
    except subprocess.CalledProcessError as exc:
        print(f"\nOllama deployment failed with exit code {exc.returncode}.")
        return

    print("\nOllama model registered successfully.")
    print("Update config/detection_config.yaml with:")
    print(f'  model: "{model_name}"')
    print('  backend: "ollama"')
    print('  base_url: "http://localhost:11434/v1"\n')

import json
from pathlib import Path

def file_exists(path):
    return Path(path).exists()

def save_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def index_by_id(rows):
    return {r["id"]: r for r in rows}

def combine_predictions(prompt_file, rag_file, tool_file):
    prompt = index_by_id(load_jsonl(prompt_file))
    rag = index_by_id(load_jsonl(rag_file))
    tool = index_by_id(load_jsonl(tool_file))
    ids = sorted(prompt.keys())
    combined = []

    for id_ in ids:
        merged = dict(prompt[id_])

        for k, v in rag[id_].items():
            if k not in merged:
                merged[k] = v
            elif k not in {"id", "label"}:
                merged[f"rag_{k}"] = v

        for k, v in tool[id_].items():
            if k not in merged:
                merged[k] = v
            elif k not in {"id", "label"}:
                merged[f"tool_{k}"] = v

        pred_prompt = prompt[id_]["pred"]
        pred_rag = rag[id_]["pred"]
        pred_tool = tool[id_]["pred"]

        final_pred = int(pred_prompt == 1 or pred_rag == 1 or pred_tool == 1)
        merged.update({
            "pred_prompt": pred_prompt,
            "pred_rag": pred_rag,
            "pred_tool": pred_tool,
            "pred": final_pred
        })

        combined.append(merged)

    return combined

def run_baseline_test():
    while True:
        choice = ask_choice(
            "BASELINES",
            [
                ("1", "Run RAGuard"),
                ("2", "Run Tool Result Parsing"),
                ("3", "Run Prompt Sentinel"),
                ("4", "Build Combined baseline"),
                ("0", "Back"),
            ],
        )

        if choice == "1":
            from models.baseline.raguard import run_raguard
            run_raguard()
        elif choice == "2":
            from models.baseline.toolparse import run_tool_result_parsing
            run_tool_result_parsing()
        elif choice == "3":
            from models.baseline.sentinel import run_prompt_channel_sentinel
            run_prompt_channel_sentinel()
        elif choice == "4":
            print("\nChecking prediction files...")
            run_baseline_test_combined_only()
        elif choice == "0":
            print("Leaving baseline menu.")
            break

        else:
            print("Invalid option. Please choose 0-4.")

def run_baseline_eval():
    from eval.eval_asr_within_baseline import run_eval_asr_for_baseline
    paths = {
        "rag": "models/baseline/raguard_predictions.jsonl",
        "tool": "models/baseline/tool_result_parsing_predictions.jsonl",
        "prompt": "models/baseline/sentinel_predictions.jsonl"
    }
    run_eval_asr_for_baseline(paths)


def confirm_data_regeneration(scope):
    print("\nThis will overwrite generated data artifacts.")
    if scope in {"split", "all"}:
        print("Affected: data/splits/train.jsonl, val.jsonl, test.jsonl")
    if scope in {"preprocess", "all"}:
        print("Affected: data/processed/train_processed.jsonl, val_processed.jsonl, test_processed.jsonl")
    print("Raw files under data/raw/ will not be changed by this menu action.")

    answer = input("Type YES to continue: ").strip()
    if answer != "YES":
        print("Cancelled.")
        return False
    return True


def run_python_module(module_name):
    command = [sys.executable, "-m", module_name]
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def run_data_split_step():
    if not confirm_data_regeneration("split"):
        return
    print_header("REGENERATE DATA SPLITS")
    run_python_module("data.data_split")
    print("\nData splits regenerated from data/raw/.")


def run_data_preprocess_step():
    if not confirm_data_regeneration("preprocess"):
        return
    print_header("REGENERATE PROCESSED DATA")
    run_python_module("data.data_preprocess")
    print("\nProcessed data regenerated from data/splits/.")


def run_full_data_regeneration():
    if not confirm_data_regeneration("all"):
        return
    print_header("FULL DATA REGENERATION")
    run_python_module("data.data_split")
    run_python_module("data.data_preprocess")
    print("\nData splits and processed files regenerated from current data/raw/.")


def run_data_preparation_menu():
    while True:
        choice = ask_choice(
            "DATA PREPARATION",
            [
                ("1", "Regenerate splits from data/raw"),
                ("2", "Regenerate processed files from data/splits"),
                ("3", "Full regenerate: raw -> splits -> processed"),
                ("0", "Back"),
            ],
        )

        try:
            if choice == "0":
                return
            if choice == "1":
                run_data_split_step()
            elif choice == "2":
                run_data_preprocess_step()
            elif choice == "3":
                run_full_data_regeneration()
        except subprocess.CalledProcessError as exc:
            print(f"\nData preparation failed with exit code {exc.returncode}.")


def run_detection_menu():
    while True:
        choice = ask_choice(
            "DEFENSE PIPELINE",
            [
                ("1", "Interactive mode picker"),
                ("2", "Run mode 10: L1 -> L2"),
                ("3", "Run mode 11: L1 -> L2 -> L3"),
                ("4", "Run mode 12: L2 -> L1 -> L3"),
                ("5", "Run mode 13: L1 -> L3"),
                ("6", "Run mode 14: L2 -> L3"),
                ("7", "Run mode 15: L2 -> L1"),
                ("8", "Run mode 31: L3 V2 only (channel-split CoT)"),
                ("9", "Run mode 32: L3 V3 only (pre-filter funnel + guided LLM)"),
                ("10", "Run mode 33: L3 V4 only (evidence-gated verifier)"),
                ("11", "Run mode 34: L3 V5 only (conservative fallback variant)"),
                ("12", "Run mode 35: L3 V6 only (compact conservative fallback)"),
                ("13", "Run mode 36: L3 V7 only (balanced evidence-gated fallback)"),
                ("14", "Run mode 37: L3 V8 only (pure LLM dual-auditor fusion)"),
                ("15", "Run mode 40: L3 V10 only (V6-equivalent base-class refactor)"),
                ("16", "Run mode 41: L3 V11 only (pure LLM-first multi-pass reviewer)"),
                ("17", "Run mode 42: L3 V12 only (original-prompt pure LLM staged reviewer)"),
                ("18", "Run mode 43: L3 O1 only (instruction extractor + auditor detector)"),
                ("19", "Run mode 44: L3 O2 only (single-call channel auditor)"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        if choice == "1":
            run_detection_pipeline()
        elif choice == "2":
            run_detection_pipeline_for_mode(10)
        elif choice == "3":
            run_detection_pipeline_for_mode(11)
        elif choice == "4":
            run_detection_pipeline_for_mode(12)
        elif choice == "5":
            run_detection_pipeline_for_mode(13)
        elif choice == "6":
            run_detection_pipeline_for_mode(14)
        elif choice == "7":
            run_detection_pipeline_for_mode(15)
        elif choice == "8":
            run_detection_pipeline_for_mode(31)
        elif choice == "9":
            run_detection_pipeline_for_mode(32)
        elif choice == "10":
            run_detection_pipeline_for_mode(33)
        elif choice == "11":
            run_detection_pipeline_for_mode(34)
        elif choice == "12":
            run_detection_pipeline_for_mode(35)
        elif choice == "13":
            run_detection_pipeline_for_mode(36)
        elif choice == "14":
            run_detection_pipeline_for_mode(37)
        elif choice == "15":
            run_detection_pipeline_for_mode(40)
        elif choice == "16":
            run_detection_pipeline_for_mode(41)
        elif choice == "17":
            run_detection_pipeline_for_mode(42)
        elif choice == "18":
            run_detection_pipeline_for_mode(43)
        elif choice == "19":
            run_detection_pipeline_for_mode(44)


def run_evaluation_menu():
    while True:
        choice = ask_choice(
            "EVALUATIONS",
            [
                ("1", "Prompt attacks without defense (LLM judge)"),
                ("2", "All attacks without defense"),
                ("3", "Defense and baseline ASR reports"),
                ("4", "Baseline-only ASR report"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        if choice == "1":
            run_without_defense_prompt_injection_eval()
        elif choice == "2":
            run_without_defense_other_eval()
        elif choice == "3":
            run_within_defense_eval()
        elif choice == "4":
            run_baseline_eval()


def run_training_menu():
    while True:
        choice = ask_choice(
            "TRAINING",
            [
                ("1", "Fine-tune Layer 1 encoder (ModernBERT)"),
                ("2", "Train Layer 1 AutoEncoder"),
                ("3", "Fine-tune Layer 2 model (ProtectAI)"),
                ("4", "Fine-tune Layer 3 local Llama 3.1 (Ollama-aligned)"),
                ("5", "Deploy fine-tuned Layer 3 model to Ollama"),
                ("6", "Find optimal Layer 1 threshold"),
                ("7", "Isolated Layer 1 retrain + high-benign Mode 11 sweep"),
                ("8", "Extract FP samples from isolation Mode 11 result"),
                ("9", "Re-evaluate trained isolation models with current Layer 3"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        if choice == "1":
            run_fine_tune_modermBERT()
        elif choice == "2":
            run_training_autoencoder()
        elif choice == "3":
            run_finetune_protectai()
        elif choice == "4":
            run_finetune_layer3_llama31()
        elif choice == "5":
            run_deploy_layer3_llama31_to_ollama()
        elif choice == "6":
            find_threshold_layer1()
        elif choice == "7":
            run_isolation_layer1_sweep()
        elif choice == "8":
            run_extract_isolation_fp_samples()
        elif choice == "9":
            run_reevaluate_isolation_models()


def run_sweeps_menu():
    while True:
        choice = ask_choice(
            "SWEEPS",
            [
                ("1", "Threshold sweep (Issue 2)"),
                ("2", "Distribution sweep (Issue 2)"),
                ("3", "High-benign distribution sweep (Issue 2, modes 11 and 12)"),
                ("4", "Layer 3 backend latency sweep (Issue 2, black-box service)"),
                ("5", "High-benign backend latency sweep (Issue 2, staged black-box service)"),
                ("6", "Layer 3 LLM backend comparison (extra models test)"),
                ("7", "Layer 3 queueing decomposition sweep (Issue 2, fixed workload)"),
                ("8", "Std testing (multi-seed L1/L2 retraining, error bars for Table 6.4)"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        if choice == "1":
            run_threshold_sweep(ask_mode("Enter mode for threshold sweep (11 or 12): "))
        elif choice == "2":
            run_distribution_sweep(mode_id=ask_mode("Enter mode for distribution sweep (11 or 12): "), fixed_threshold=0.14)
        elif choice == "3":
            run_high_benign_distribution_sweep()
        elif choice == "4":
            run_layer3_backend_latency_sweep()
        elif choice == "5":
            from experiments.high_benign_backend_latency_sweep import (
                run_high_benign_backend_latency_sweep,
            )

            run_high_benign_backend_latency_sweep()
        elif choice == "6":
            from experiments.llm_backend_comparison import run_llm_backend_comparison

            run_llm_backend_comparison()
        elif choice == "7":
            from experiments.layer3_queueing_decomposition import (
                run_layer3_queueing_decomposition_sweep,
            )

            run_layer3_queueing_decomposition_sweep()
        elif choice == "8":
            run_std_testing_menu()


def run_std_testing_menu():
    while True:
        choice = ask_choice(
            "STD TESTING (multi-seed L1/L2 + error bars for Table 6.4)",
            [
                ("1", "Phase 1: Train Layer 1 + Layer 2 with N different seeds"),
                ("2", "Phase 2: Evaluate all Table 6.4 modes per trial + aggregate mean/std"),
                ("3", "Aggregate: recompute summary.json/summary.csv from existing results"),
                ("4", "Generate figures: Issue 1 precision/recall/latency chart (mean±std)"),
                ("0", "Back"),
            ],
        )
        if choice == "0":
            return
        if choice == "1":
            from experiments.std_testing.train import run_std_testing_train

            run_std_testing_train()
        elif choice == "2":
            from experiments.std_testing.evaluate import run_std_testing_evaluate

            run_std_testing_evaluate()
        elif choice == "3":
            from experiments.std_testing.evaluate import run_std_testing_aggregate

            run_std_testing_aggregate()
        elif choice == "4":
            from experiments.std_testing.figures import run_std_testing_generate_figures

            run_std_testing_generate_figures()


def run_figures_menu():
    while True:
        choice = ask_choice(
            "FIGURES",
            [
                ("1", "Generate all figures"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        if choice == "1":
            run_figure_generation()


def run_guided_workflows_menu():
    while True:
        choice = ask_choice(
            "GUIDED WORKFLOWS",
            [
                ("1", "Build full 3-layer result (mode 11)"),
                ("2", "Refresh baseline predictions + combined"),
                ("3", "Refresh baseline evaluation after baseline changes"),
                ("4", "Generate Issue 3 ASR line chart from ready outputs"),
                ("5", "Generate Issue 3 ASR line chart from scratch"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        if choice == "1":
            print("\nWorkflow: 1 -> mode 11")
            run_detection_pipeline_for_mode(11)
        elif choice == "2":
            print("\nWorkflow: baseline prompt + rag + tool + combined")
            from models.baseline.raguard import run_raguard
            from models.baseline.toolparse import run_tool_result_parsing
            from models.baseline.sentinel import run_prompt_channel_sentinel

            run_raguard()
            run_tool_result_parsing()
            run_prompt_channel_sentinel()
            combined = combine_predictions(
                "models/baseline/sentinel_predictions.jsonl",
                "models/baseline/raguard_predictions.jsonl",
                "models/baseline/tool_result_parsing_predictions.jsonl",
            )
            save_jsonl("models/baseline/combined_predictions.jsonl", combined)
            print("\nCombined predictions saved to:")
            print("models/baseline/combined_predictions.jsonl")
        elif choice == "3":
            print("\nWorkflow: baseline eval only")
            run_baseline_eval()
        elif choice == "4":
            print("\nWorkflow: combined baseline -> defense eval -> figures")
            run_baseline_test_combined_only()
            run_within_defense_eval()
            run_figure_generation()


def run_documentation_menu():
    while True:
        choice = ask_choice(
            "DOCUMENTATION & ABOUT",
            [
                ("1", "Read introduction"),
                ("2", "Read workflow guide"),
                ("3", "Read about Le Minh Bang"),
                ("4", "Read quick reference"),
                ("0", "Back"),
            ],
        )

        if choice == "0":
            return
        elif choice == "1":
            view_document_lines(build_intro_document(), "INTRODUCTION | AGENT DEFENSE")
        elif choice == "2":
            view_document_lines(build_workflow_document(), "WORKFLOW GUIDE")
        elif choice == "3":
            view_document_lines(build_about_document(), "ABOUT LE MINH BANG")
        elif choice == "4":
            view_document_lines(build_quick_reference_document(), "QUICK REFERENCE")
        elif choice == "5":
            print("\nWorkflow: no-defense eval -> mode 11 -> combined baseline -> defense eval -> figures")
            run_without_defense_prompt_injection_eval()
            run_without_defense_other_eval()
            run_detection_pipeline_for_mode(11)
            run_baseline_test_combined_only()
            run_within_defense_eval()
            run_figure_generation()


def run_baseline_test_combined_only():
    rag_pred = "models/baseline/raguard_predictions.jsonl"
    tool_pred = "models/baseline/tool_result_parsing_predictions.jsonl"
    prompt_pred = "models/baseline/sentinel_predictions.jsonl"
    combined_pred = "models/baseline/combined_predictions.jsonl"

    missing = []
    if not file_exists(rag_pred):
        missing.append("raguard")
    if not file_exists(tool_pred):
        missing.append("tool")
    if not file_exists(prompt_pred):
        missing.append("prompt")

    if missing:
        print("Missing prediction files:", ", ".join(missing))
        print("Running required baseline detectors first...\n")

        if "raguard" in missing:
            from models.baseline.raguard import run_raguard
            run_raguard()
        if "tool" in missing:
            from models.baseline.toolparse import run_tool_result_parsing
            run_tool_result_parsing()
        if "prompt" in missing:
            from models.baseline.sentinel import run_prompt_channel_sentinel
            run_prompt_channel_sentinel()

    combined = combine_predictions(prompt_pred, rag_pred, tool_pred)
    save_jsonl(combined_pred, combined)
    print("\nCombined predictions saved to:")
    print(combined_pred)
def main():
    print_startup_brand()

    while True:
        choice = ask_choice(
            "AGENT DEFENSE SYSTEM",
            [
                ("1", "Defense pipeline"),
                ("2", "Baselines"),
                ("3", "Evaluations"),
                ("4", "Figures"),
                ("5", "Training and threshold tools"),
                ("6", "Sweeps"),
                ("7", "Guided workflows"),
                ("8", "Documentation and about"),
                ("9", "Backup current results"),
                ("10", "Data preparation"),
                ("0", "Exit"),
            ],
        )

        if choice == "0":
            print("Exiting...")
            break
        elif choice == "1":
            run_detection_menu()
        elif choice == "2":
            run_baseline_test()
        elif choice == "3":
            run_evaluation_menu()
        elif choice == "4":
            run_figures_menu()
        elif choice == "5":
            run_training_menu()
        elif choice == "6":
            run_sweeps_menu()
        elif choice == "7":
            run_guided_workflows_menu()
        elif choice == "8":
            run_documentation_menu()
        elif choice == "9":
            run_backup_current_results()
        elif choice == "10":
            run_data_preparation_menu()
        print("\nReturning to main menu....\n")


if __name__ == "__main__":
    main()
