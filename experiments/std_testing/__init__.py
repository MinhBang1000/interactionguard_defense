"""Std Testing: multi-seed Layer 1 / Layer 2 retraining + error bars for Table 6.4.

Phase 1 (train.py): retrains Layer 1 (ModernBERT + AutoEncoder) and Layer 2
(DeBERTa) with N different random seeds, keeping the fixed train/val/test
split and every other hyperparameter identical to the existing single-run
training options in main.py's "Training and threshold tools" menu.

Phase 2 (evaluate.py): re-runs the 9 modes that make up the thesis's
Table 6.4 (config/test_modes.yaml modes 1, 2, 3, 10, 11, 12, 13, 14, 15) once
per trial, with Layer 3 pinned to the O2 variant (the one that produced the
existing results/mode_*_results.json reference numbers), then aggregates
mean +/- std per mode/metric.

Both phases write only under models/std_testing/, config/std_testing/, and
results/std_testing/ -- the production checkpoints and reference results are
never touched.
"""
