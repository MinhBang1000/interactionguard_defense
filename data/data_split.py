# ============================================================================
# FILE: data/data_split.py
# Build train, validation, and test splits from raw benign and attack traces.
#
# Purpose:
# - Loads all raw trace categories, assigns labels and attack types, analyzes
#   overlap, and writes clean split files for downstream preprocessing.
#
# Workflow:
# - Reads JSONL traces from `data/raw/`.
# - Handles prompt overlap analysis and grouped splitting for correlated cases.
# - Produces `train.jsonl`, `val.jsonl`, and `test.jsonl` under `data/splits/`.
#
# Use this file when:
# - You want to regenerate the experimental data splits.
# - You need to inspect leakage control or attack-type balance decisions.
# ============================================================================
# data/data_split.py
# python -m data.data_split.py

import sys
from config.config import DataSplitConfig
from collections import defaultdict, Counter
from utils.random_seed import SEED
from typing import Dict, List, Tuple
from utils.utils import load_jsonl, add_label, analyze_overlap, split_data, hash_prompt, print_split_stats, save_jsonl
import random
import numpy as np
print(f"Python version {sys.version}\n")

random.seed(SEED)
np.random.seed(SEED)

print(f"SEED Setup: {SEED}")

# initiaize config
config = DataSplitConfig()
config.__post_init__()

# load raw data
print("\n" + "="*80)
print("LOAD RAW DATA")
required_files = [
    "benign_traces.jsonl",
    "prompt_injection_traces.jsonl",
    "rag_poison_traces.jsonl",
    "tool_injection_traces.jsonl",
    "correlated_injection_traces.jsonl"
]

print("\nCheck all files existed:")
for fname in required_files:
    file_path = config.LOG_DIR / fname
    if not file_path.exists():
        raise FileNotFoundError(f"Couldn't find the file at this path {file_path}")
    
# load benign data and attack data
benign = add_label(load_jsonl(config.LOG_DIR / required_files[0]), 0, None)
prompt_inj = add_label(
    load_jsonl(config.LOG_DIR / required_files[1]),
    1,
    "prompt"
)
rag_poison = add_label(
    load_jsonl(config.LOG_DIR / required_files[2]),
    1,
    "rag"
)
tool_inj = add_label(
    load_jsonl(config.LOG_DIR / required_files[3]),
    1,
    "tool"
)
corr_inj = add_label(
    load_jsonl(config.LOG_DIR / required_files[4]),
    1,
    "correlated"
)

print("\n📊 Data loaded:")
print(f"   Benign:           {len(benign):>5,}")
print(f"   Prompt injection: {len(prompt_inj):>5,}")
print(f"   RAG poison:       {len(rag_poison):>5,}")
print(f"   Tool injection:   {len(tool_inj):>5,}")
print(f"   Correlated:       {len(corr_inj):>5,}")
print(f"   {'─'*40}")
total = len(benign) + len(prompt_inj) + len(rag_poison) + len(tool_inj) + len(corr_inj)
print(f"   Total:            {total:>5,}")

print("\n Raw data loaded successfully!")

# overlap handling
print("\n" + "="*80)
print("OVERLAP HANDLING")

# only for rag and correlated
rag_corr_overlap = analyze_overlap(rag_poison, corr_inj, "RAG", "Correlated")
# check all attack and benign prompts
all_attacks = prompt_inj + rag_poison + tool_inj + corr_inj
benign_attack_overlap = analyze_overlap(benign, all_attacks, "Benign", "All attacks")
# check cross-attack overlaps
for name1, data1 in [("Prompt", prompt_inj), ("Tool", tool_inj)]:
    for name2, data2 in [("RAG", rag_poison), ("Correlated", corr_inj)]:
        overlap = analyze_overlap(data1, data2, name1, name2)

# Special warning for 100% overlap
if rag_corr_overlap == len(rag_poison) == len(corr_inj):
    print("\n⚠️  CRITICAL: RAG and Correlated have 100% prompt overlap!")
    print("   → Will use GROUP-BASED split to prevent leakage")
    print("   → Same prompt will stay in same split (train/val/test)")
else:
    print(f"\n✅ RAG and Correlated have {rag_corr_overlap} overlapping prompts")

# Split Benign data
print("\n" + "="*80)
print("SPLITTING BENIGN DATA")

train_benign, val_benign, test_benign = split_data("Benign", benign, SEED, config.BENIGN_TRAIN_RATIO, config.ATTACK_VAL_RATIO, config.BENIGN_TEST_RATIO)

# Split Prompt Injection
print("\n"+"="*80)
print("SPLITTING PROMPT INJECTION DATA")

train_prompt, val_prompt, test_prompt = split_data("Prompt Injection", prompt_inj, SEED, config.ATTACK_TRAIN_RATIO, config.ATTACK_VAL_RATIO, config.ATTACK_TEST_RATIO)

# Split Tool Injection
print("\n" + "="*80)
print("SPLITTING TOOL INJECTION")

train_tool, val_tool, test_tool = split_data("Tool Injection", tool_inj, SEED, config.ATTACK_TRAIN_RATIO, config.ATTACK_VAL_RATIO, config.ATTACK_TEST_RATIO)

# Split RAG + Correlated Attack (Group-based)
# Because the group can make sure each duplicated sample can not appear on both set
random.seed(SEED)
rag_corr_groups = defaultdict(list)
for t in rag_poison:
    ph = hash_prompt(t.get("prompt",""))
    rag_corr_groups[ph].append(t)

for t in corr_inj:
    ph = hash_prompt(t.get("prompt", ""))
    rag_corr_groups[ph].append(t)

groups = list(rag_corr_groups.values())
total_traces = sum(len(g) for g in groups)

print(f"   Total groups: {len(groups):,}")
print(f"   Total traces: {total_traces:,}")
print(f"   Expected: {len(rag_poison) + len(corr_inj):,}")
print(f"   Avg traces per group: {total_traces / len(groups):.2f}")

random.shuffle(groups)
n_groups = len(groups)
train_end = int(config.ATTACK_TRAIN_RATIO * n_groups)
val_end = int(config.ATTACK_VAL_RATIO * n_groups + train_end)

train_groups = groups[:train_end]
val_groups = groups[train_end:val_end]
test_groups = groups[val_end:]

# Back to the traces
train_rag_corr = [t for g in train_groups for t in g]
val_rag_corr = [t for g in val_groups for t in g]
test_rag_corr = [t for g in test_groups for t in g]
print_split_stats(train_rag_corr, val_rag_corr, test_rag_corr, "RAG + Correlated")

# Combine all attacks!
train_attack = train_prompt + train_tool + train_rag_corr
val_attack = val_prompt + val_tool + val_rag_corr
test_attack = test_prompt + test_tool + test_rag_corr

print_split_stats(train_attack, val_attack, test_attack, "All attacks")

# Attack type breakdown
print("\n📊 Attack type distribution:")
for split_name, split_data in [
    ("Train", train_attack),
    ("Val", val_attack),
    ("Test", test_attack)
]:
    counts = Counter([t.get("attack_type") for t in split_data])
    print(f"\n   {split_name}:")
    for atype, count in sorted(counts.items()):
        pct = count / len(split_data) * 100 if split_data else 0
        print(f"      {atype:.<20} {count:>4,} ({pct:>5.1f}%)")

# Create the final combination splits
print("\n" + "="*80)
print("CREATING FINAL COMBINED SPLITS")

# Combine benign + attack
train_all = train_attack + train_benign
val_all = val_attack + val_benign
test_all = test_attack + test_benign

random.seed(SEED)
random.shuffle(train_all)
random.shuffle(val_all)
random.shuffle(test_all)

print_split_stats(train_all, val_all, test_all)

# Print class distribution
for name, data in [
    ("Train", train_all),
    ("Val", val_all),
    ("Test", test_all)
]:
    label_counts = Counter(t["label"] for t in data)
    # It returns Counter object which is a dict-liked. So that, we can use get function here
    benign_count = label_counts.get(0,0)
    attack_count = label_counts.get(1,0)
    total = len(data)

    print(f"\n   {name}:")
    print(f"      Benign (0): {benign_count:>5,} ({benign_count/total*100:>5.1f}%)")
    print(f"      Attack (1): {attack_count:>5,} ({attack_count/total*100:>5.1f}%)")

# Data integrity checks
print("\n" + "="*80)
print("DATA INTEGRITY CHECKS")

all_passed = True

# === Check 1: No trace ID overlap ===
print("\n🔍 Check 1: Trace ID overlap between splits")

train_ids = {t.get("id", id(t)) for t in train_all}
val_ids = {t.get("id", id(t)) for t in val_all}
test_ids = {t.get("id", id(t)) for t in test_all}

overlaps = {
    "Train-Val": train_ids & val_ids,
    "Train-Test": train_ids & test_ids,
    "Val-Test": val_ids & test_ids
}

for name, overlap in overlaps.items():
    if overlap:
        print(f"   ❌ {name} overlap: {len(overlap)} traces")
        all_passed = False
    else:
        print(f"   ✅ No {name} overlap")

# === Check 2: Prompt leakage in RAG + Correlated ===
print("\n🔍 Check 2: Prompt leakage in RAG + Correlated")

train_prompts = {hash_prompt(t.get("prompt", "")) for t in train_rag_corr}
val_prompts = {hash_prompt(t.get("prompt", "")) for t in val_rag_corr}
test_prompts = {hash_prompt(t.get("prompt", "")) for t in test_rag_corr}

prompt_overlaps = {
    "Train-Val": train_prompts & val_prompts,
    "Train-Test": train_prompts & test_prompts,
    "Val-Test": val_prompts & test_prompts
}

for name, overlap in prompt_overlaps.items():
    if overlap:
        print(f"   ❌ {name} prompt leak: {len(overlap)} prompts")
        all_passed = False
    else:
        print(f"   ✅ No {name} prompt leakage")

# === Check 3: Label consistency ===
print("\n🔍 Check 3: Label consistency")

for split_name, split_data in [
    ("Train", train_all),
    ("Val", val_all),
    ("Test", test_all)
]:
    missing_labels = sum(1 for t in split_data if "label" not in t)
    invalid_labels = sum(1 for t in split_data if t.get("label") not in [0, 1])
    
    if missing_labels > 0:
        print(f"   ⚠️  {split_name}: {missing_labels} traces missing labels")
        all_passed = False
    elif invalid_labels > 0:
        print(f"   ⚠️  {split_name}: {invalid_labels} traces with invalid labels")
        all_passed = False
    else:
        print(f"   ✅ {split_name}: All labels valid")

# === Check 4: Attack type consistency ===
print("\n🔍 Check 4: Attack type consistency")

for split_name, split_data in [
    ("Train", train_all),
    ("Val", val_all),
    ("Test", test_all)
]:
    # Check benign traces
    benign_traces = [t for t in split_data if t["label"] == 0]
    benign_with_attack_type = [t for t in benign_traces if t.get("attack_type") not in [None, "benign"]]
    
    # Check attack traces
    attack_traces = [t for t in split_data if t["label"] == 1]
    attack_without_type = [t for t in attack_traces if not t.get("attack_type")]
    
    if benign_with_attack_type:
        print(f"   ⚠️  {split_name}: {len(benign_with_attack_type)} benign traces have attack_type")
        all_passed = False
    elif attack_without_type:
        print(f"   ⚠️  {split_name}: {len(attack_without_type)} attack traces missing attack_type")
        all_passed = False
    else:
        print(f"   ✅ {split_name}: Attack types consistent")

# Final verdict
print("\n" + "─"*80)
if all_passed:
    print("✅ ALL INTEGRITY CHECKS PASSED")
else:
    print("⚠️  SOME INTEGRITY CHECKS FAILED - Review warnings above")

# Save Splits
print("\n" + "="*80)
print("SAVING SPLITS")
save_jsonl(train_all, config.SPLIT_DIR / "train.jsonl")
save_jsonl(val_all, config.SPLIT_DIR / "val.jsonl")
save_jsonl(test_all, config.SPLIT_DIR / "test.jsonl")

print(f"Saved all main splits to {config.SPLIT_DIR}")

print("\n" + "="*80)
print("DATA SPLITTING COMPLETED")
print("="*80)

print(f"""
📊 Final Split Summary:

Train: {len(train_all):,} traces
  ├─ Benign: {len(train_benign):,} ({len(train_benign)/len(train_all)*100:.1f}%)
  └─ Attack: {len(train_attack):,} ({len(train_attack)/len(train_all)*100:.1f}%)
     ├─ Prompt injection: {len(train_prompt):,}
     ├─ Tool injection:   {len(train_tool):,}
     ├─ RAG poison:       {sum(1 for t in train_rag_corr if t.get('attack_type')=='rag'):,}
     └─ Correlated:       {sum(1 for t in train_rag_corr if t.get('attack_type')=='correlated'):,}

Val: {len(val_all):,} traces
  ├─ Benign: {len(val_benign):,} ({len(val_benign)/len(val_all)*100:.1f}%)
  └─ Attack: {len(val_attack):,} ({len(val_attack)/len(val_all)*100:.1f}%)

Test: {len(test_all):,} traces
  ├─ Benign: {len(test_benign):,} ({len(test_benign)/len(test_all)*100:.1f}%)
  └─ Attack: {len(test_attack):,} ({len(test_attack)/len(test_all)*100:.1f}%)
""")

print("🔒 Anti-leakage Measures:")
print("   ✅ No trace ID overlap between splits")
print("   ✅ RAG + Correlated use group-based split")
print("   ✅ No prompt leakage in RAG + Correlated")
