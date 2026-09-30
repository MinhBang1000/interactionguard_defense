# ============================================================================
# FILE: data/data_preprocess.py
# Convert split trace files into processed sequence samples for model use.
#
# Purpose:
# - Turns raw split traces into normalized sequence samples with special tokens
#   and optional multi-step prefix expansion.
#
# Workflow:
# - Reads split files from `data/splits/`.
# - Uses preprocessing helpers from `utils/utils.py`.
# - Writes processed JSONL files into `data/processed/`.
#
# Use this file when:
# - You want to refresh processed training/evaluation data after modifying
#   trace formatting, prefix logic, or split files.
# ============================================================================
# data/data_preprocess.py
# python -m data.data_preprocess.py (Locate at root dir)

import sys
print(f"Python version: {sys.version}")
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional
from config.config import DataPreprocessConfig
from utils.utils import process_file

import numpy as np
from tqdm.auto import tqdm

def main():
    config = DataPreprocessConfig()
    config.__post_init__()
    print("Configuration successful!")
    
    print("\n"+"="*80)
    print("STARTING PREPROCESSING")
    print("="*80)

    all_stats = []
    for input_file in config.INPUT_FILES:
        split_name = input_file.replace(".jsonl","")
        input_path = config.SPLIT_DIR / input_file
        output_file = f"{split_name}_processed.jsonl"
        output_path = config.PROCESSED_DIR / output_file

        generate_multi_step = config.GENERATE_MULTI_STEP.get(split_name, True)

        stats = process_file(
            input_path=input_path,
            output_path=output_path,
            generate_multi_step=generate_multi_step,
            special_tokens=config.SPECIAL_TOKENS
        )

        all_stats.append(stats)

    print("PREPROCESSING COMPLETED")


if __name__ == "__main__":
    main()
