"""Train a reproducible encoder on prepared data, without reading the test set."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import time

from research_runner.progress import emit, trainer_callback


def configure_loss_normalization(model):
    # ModernBERT's mean CE does not consume num_items_in_batch (Transformers 5).
    model.accepts_loss_kwargs = False
    return model


def run(train_percent=None, *, output_dir=None, smoke=False, seed=None):
    import numpy as np
    import torch
    from datasets import Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
                              TrainingArguments, TrainerCallback, set_seed)
    from research_runner.implementation.training.settings import ModernBERTConfig
    from research_runner.implementation.data.data_preprocess import ROOT, sha256, write_json
    from research_runner.implementation.data.prepare_training import load_prepared
    from research_runner.implementation.training.gate_common import provenance, timestamp

    config = ModernBERTConfig()
    run_seed = config.SEED if seed is None else seed
    set_seed(run_seed)
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for this long-context encoder run")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This configuration requires BF16 support")
    ratio = config.TRAIN_RATIO if train_percent is None else train_percent / 100
    if not 0 < ratio <= 1:
        raise ValueError("train_percent must be between 1 and 100")
    output = Path(output_dir or config.OUTPUT_DIR)
    if (output / "training_config.json").exists():
        raise ValueError(f"Refusing to overwrite a trained encoder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    train = load_prepared("encoder", "train", output_dir=config.DATA_DIR.parent)
    val = load_prepared("encoder", "val", output_dir=config.DATA_DIR.parent)
    order = np.random.default_rng(run_seed).permutation(len(train))
    train = [train[i] for i in order[:max(1, int(len(train)*ratio))]]
    emit('encoder.tokenizer', message=f'Loading tokenizer; seed={run_seed}.', force=True)
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME, local_files_only=True)
    tokenizer.padding_side, tokenizer.truncation_side = "right", "right"
    settings = {"model_name": config.MODEL_NAME, "max_length": config.MAX_LENGTH,
                "trust_remote_code": False, "pooling": "cls", "classifier_pooling": "cls",
                "padding": "dynamic", "truncation_side": "right", "precision": "bf16",
                "group_by_length": True,
                "embedding_precision": "bf16", "embedding_batch_size": 1,
                "attention_implementation": "sdpa", "gradient_checkpointing": True,
                "gradient_accumulation_loss_normalized_by_trainer": True,
                "epochs": config.EPOCHS, "learning_rate": config.LR,
                "batch_size": config.BATCH_SIZE, "gradient_accumulation": config.GRAD_ACCUM,
                "effective_batch_size": config.BATCH_SIZE*config.GRAD_ACCUM,
                "weight_decay": config.WEIGHT_DECAY, "warmup_ratio": config.WARMUP_RATIO,
                "max_grad_norm": config.MAX_GRAD_NORM, "seed": run_seed,
                "train_ratio": ratio, "train_samples": len(train), "val_samples": len(val),
                "eval_strategy": "epoch", "checkpoint_selection": "validation_f1",
                "early_stopping_patience": config.EARLY_STOP_PATIENCE,
                "save_total_limit": 2, "dataloader_num_workers": 0,
                "prepared_manifest_sha256": sha256(ROOT / "data/training/manifest.json"),
                "source_sha256": sha256(Path(__file__)), "provenance": provenance(),
                "started_at_utc": timestamp(), "smoke_test": smoke}
    write_json(output / "run_config.json", settings)
    write_json(output / "sample_ids.json", {"train": [r["id"] for r in train], "val": [r["id"] for r in val]})

    def dataset(rows):
        data = Dataset.from_dict({"text": [r["sequence"] for r in rows], "labels": [r["label"] for r in rows]})
        return data.map(lambda examples: tokenizer(examples["text"], truncation=True,
                                                    max_length=config.MAX_LENGTH),
                        batched=True, remove_columns=["text"])

    emit('encoder.tokenization', 0, 2, unit='splits', message=f'Train={len(train)}; validation={len(val)}.', force=True)
    train_ds = dataset(train)
    emit('encoder.tokenization', 1, 2, unit='splits', force=True)
    val_ds = dataset(val)
    emit('encoder.tokenization', 2, 2, unit='splits', status='completed', force=True)
    emit('encoder.model_load', message='Loading ModernBERT weights.', force=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.MODEL_NAME, num_labels=2, local_files_only=True,
        classifier_pooling="cls", reference_compile=False, attn_implementation="sdpa")
    # ModernBERT computes mean CE but ignores num_items_in_batch in Transformers 5.
    # Tell Trainer to divide by accumulation steps rather than counting loss twice.
    configure_loss_normalization(model)
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    if smoke:
        emit('encoder.smoke', message='Testing forward/backward at the longest training sequence.', force=True)
        model.to("cuda:0").train()
        model.gradient_checkpointing_enable()
        index = max(range(len(train_ds)), key=lambda i: len(train_ds[i]["input_ids"]))
        batch = {k: v.to("cuda:0") for k, v in collator([train_ds[index]]*config.BATCH_SIZE).items()}
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite smoke-test loss")
        torch.cuda.synchronize()
        report = {"loss": float(loss.detach()), "micro_batch_size": config.BATCH_SIZE,
                  "sequence_length": batch["input_ids"].shape[1],
                  "seconds": time.perf_counter()-start, "max_allocated_gib": torch.cuda.max_memory_allocated()/2**30,
                  "status": "forward_backward_passed"}
        write_json(output / "smoke_test.json", report)
        print(report, flush=True)
        emit('encoder.smoke', status='completed', message='Forward/backward smoke test passed.', force=True)
        return report

    def compute_metrics(prediction):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        preds = np.argmax(prediction.predictions, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(prediction.label_ids, preds, average="binary", zero_division=0)
        return {"accuracy": accuracy_score(prediction.label_ids, preds), "precision": precision, "recall": recall, "f1": f1}

    args = TrainingArguments(
        output_dir=str(output / "checkpoints"), num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE, per_device_eval_batch_size=1,
        gradient_accumulation_steps=config.GRAD_ACCUM, learning_rate=config.LR,
        weight_decay=config.WEIGHT_DECAY, warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM, bf16=True, fp16=False,
        gradient_checkpointing=True, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=2, load_best_model_at_end=True, metric_for_best_model="f1",
        greater_is_better=True, logging_steps=25, report_to="none",
        seed=run_seed, data_seed=run_seed, dataloader_num_workers=0, group_by_length=True,
        dataloader_pin_memory=True, disable_tqdm=True)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                      processing_class=tokenizer, data_collator=collator, compute_metrics=compute_metrics,
                      callbacks=[EarlyStoppingCallback(early_stopping_patience=config.EARLY_STOP_PATIENCE),
                                 trainer_callback(TrainerCallback)])
    trainer.train()
    validation = trainer.evaluate()
    emit('encoder.save', message='Saving selected encoder, tokenizer and training history.', force=True)
    trainer.save_model(output)
    tokenizer.save_pretrained(output)
    settings.update({"finished_at_utc": timestamp(), "validation": validation,
                     "best_checkpoint": trainer.state.best_model_checkpoint,
                     "best_validation_f1": trainer.state.best_metric,
                     "base_model_revision": getattr(model.config, "_commit_hash", None),
                     "model_config": model.config.to_dict(),
                     "tokenizer_sha256": sha256(output / "tokenizer.json"),
                     "training_arguments": args.to_dict()})
    write_json(output / "training_config.json", settings)
    write_json(output / "training_history.json", {"log_history": trainer.state.log_history})
    print(f"ENCODER COMPLETE {output}; validation={validation}", flush=True)
    emit('encoder.save', status='completed', message=str(output), force=True)
    return settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_percent", type=int, default=100)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=os.environ.get("CUDA_DEVICE_TRAIN", os.environ.get("CUDA_DEVICE_ML", "0")))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["CUDA_DEVICE_ML"] = args.device
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from research_runner.implementation.training.gate_runtime import gpu_preflight
    gpu_preflight(args.device)
    if args.output_dir is None:
        from datetime import datetime, timezone
        from research_runner.implementation.data.data_preprocess import ROOT
        args.output_dir = ROOT / "results/research_runner/training" / datetime.now(timezone.utc).strftime("encoder_%Y%m%d_%H%M%S")
    run(args.train_percent, output_dir=args.output_dir, smoke=args.smoke, seed=args.seed)


if __name__ == "__main__":
    main()
