"""
Std Testing: seed-override training wrappers.

Runs one Layer 1 / Layer 2 training component with a different random seed
and a different output directory than the shared production checkpoints,
WITHOUT editing config/config.py or training/*.py. Each component is meant to
be invoked as its own subprocess (see experiments/std_testing/train.py), one
subprocess per component per trial, for GPU-memory isolation -- the same
subprocess pattern main.py already uses for the original single-run training
options (run_fine_tune_modermBERT / run_training_autoencoder /
run_finetune_protectai).

Why this is safe to do without touching the original files:
- ModernBERTConfig.SEED / OUTPUT_DIR and Layer2ProtectAIConfig.SEED /
  OUTPUT_DIR are plain class attributes, never reassigned inside __init__,
  and both training scripts instantiate `config = XConfig()` *inside* their
  run() function -- so patching the class attribute right before calling
  run() is enough; run() picks up the patched values when it constructs its
  own instance.
- AutoEncoderConfig is different: its module-level `config` object is built
  at import time (training/autoencoder_training.py instantiates it and reads
  embeddings from config.FINETUNED_MODEL_PATH as import-time side effects),
  and its __init__ hardcodes `self.SEED = 42`. A pre-import class-level SEED
  patch would be silently overwritten by that hardcoded line. Instead this
  wrapper patches the class-level MODEL_DIR / FINETUNED_MODEL_PATH (both are
  plain class attributes, never reassigned in __init__, so a pre-import patch
  sticks) so embeddings are extracted from *this trial's* ModernBERT
  checkpoint and the trained autoencoder is saved to *this trial's* directory
  -- then re-seeds the global torch/numpy RNG state right before calling
  run(), which is when the actual randomness (model weight init, DataLoader
  shuffle order) happens.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _run_modernbert(seed: int, output_dir: Path, train_percent: int) -> None:
    from config.config import ModernBERTConfig

    ModernBERTConfig.SEED = seed
    ModernBERTConfig.OUTPUT_DIR = output_dir

    from training.modernbert_finetune import run

    run(train_percent=train_percent)


def _run_deberta(seed: int, output_dir: Path, train_percent: int) -> None:
    from config.config import Layer2ProtectAIConfig

    Layer2ProtectAIConfig.SEED = seed
    Layer2ProtectAIConfig.OUTPUT_DIR = output_dir

    from training.protectai_deberta_finetune import run

    run(train_percent=train_percent)


def _run_autoencoder(seed: int, output_dir: Path, modernbert_dir: Path) -> None:
    from config.config import AutoEncoderConfig

    AutoEncoderConfig.MODEL_DIR = output_dir
    AutoEncoderConfig.FINETUNED_MODEL_PATH = modernbert_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Importing this module runs its Stage-2 data loading and embedding
    # extraction as import-time side effects, using the patched
    # FINETUNED_MODEL_PATH above.
    import training.autoencoder_training as ae_mod
    import torch
    import numpy as np

    # The module seeded the RNG with the hardcoded default (42) as an
    # import-time side effect; re-seed now, right before the actual
    # randomness (model init, DataLoader shuffling) happens inside run().
    torch.manual_seed(seed)
    np.random.seed(seed)

    ae_mod.run()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Std Testing seed-override training wrapper (subprocess entrypoint)."
    )
    parser.add_argument("--component", required=True, choices=["modernbert", "autoencoder", "deberta"])
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--train_percent", type=int, default=100)
    parser.add_argument(
        "--modernbert_dir",
        type=Path,
        default=None,
        help="Required for --component autoencoder: this trial's ModernBERT checkpoint dir.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.component == "modernbert":
        _run_modernbert(args.seed, args.output_dir, args.train_percent)
    elif args.component == "deberta":
        _run_deberta(args.seed, args.output_dir, args.train_percent)
    elif args.component == "autoencoder":
        if args.modernbert_dir is None:
            raise ValueError("--modernbert_dir is required for --component autoencoder")
        _run_autoencoder(args.seed, args.output_dir, args.modernbert_dir)


if __name__ == "__main__":
    main()
