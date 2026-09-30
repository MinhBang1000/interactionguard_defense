# ============================================================================
# FILE: training/llama31_ollama_finetune.py
# Supervised fine-tuning pipeline for the local Layer 3 Llama 3.1 8B model.
#
# Purpose:
# - Turns `train_processed.jsonl` / `val_processed.jsonl` into an instruction-
#   tuning dataset aligned with the Layer 3 semantic-audit task.
# - Fine-tunes a Llama 3.1 style causal LM and exports artifacts that are easy
#   to reuse from both Hugging Face tooling and Ollama.
#
# Workflow:
# - Reads processed traces and converts them into prompt/JSON-answer pairs.
# - Supports train subset control from 1-100 percent with stratified sampling.
# - Uses QLoRA 4-bit for memory-efficient fine-tuning, then merges the learned
#   adapter back into a standard export for downstream Ollama deployment.
# - Saves the model, tokenizer, run metadata, preview datasets, and an Ollama
#   Modelfile scaffold inside `models/layer3/`.
#
# Use this file when:
# - You want a local Layer 3 model adapted to this project's processed traces.
# - You want one command path from `main.py` instead of manual training steps.
# ============================================================================

import json
import math
import os
import random
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES",
    os.getenv("CUDA_DEVICE_LLM", os.getenv("CUDA_DEVICE_ML", "0")),
)

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
)

from config.config import Layer3Llama31SFTConfig
from models.detectors.layer3_detector import Layer3Detector
from utils.utils import load_jsonl

try:
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

try:
    import bitsandbytes  # noqa: F401
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


SYSTEM_PROMPT = Layer3Detector._build_system_prompt(None).strip()


def _bool_to_json(value: bool) -> str:
    return "true" if value else "false"


def _build_reason(sample: Dict) -> str:
    if int(sample["label"]) == 0:
        return "No manipulation detected."

    attack_type = sample.get("attack_type") or "unknown"
    reason_map = {
        "prompt": "Prompt override detected.",
        "rag": "Retrieved instruction redirect detected.",
        "tool": "Tool redirection detected.",
        "correlated": "Correlated manipulation detected.",
    }
    return reason_map.get(attack_type, "Manipulative instruction detected.")


def _build_target_json(sample: Dict) -> str:
    is_malicious = bool(int(sample["label"]))
    confidence = 0.98 if is_malicious else 0.02
    return (
        "{"
        f"\"is_malicious\":{_bool_to_json(is_malicious)},"
        f"\"confidence\":{confidence:.2f},"
        f"\"reason\":{json.dumps(_build_reason(sample), ensure_ascii=False)}"
        "}"
    )


def _build_user_prompt(sample: Dict) -> str:
    return sample["sequence"].strip()


def _apply_chat_template(tokenizer, messages: List[Dict], add_generation_prompt: bool) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )

    rendered = []
    for msg in messages:
        rendered.append(f"{msg['role'].upper()}:\n{msg['content'].strip()}\n")
    if add_generation_prompt:
        rendered.append("ASSISTANT:\n")
    return "\n".join(rendered)


def _stratified_subset(samples: List[Dict], percent: int, seed: int) -> List[Dict]:
    if percent >= 100:
        return list(samples)

    grouped: Dict[int, List[Dict]] = {}
    for sample in samples:
        grouped.setdefault(int(sample["label"]), []).append(sample)

    rng = random.Random(seed)
    subset = []
    for label, rows in grouped.items():
        rows = list(rows)
        rng.shuffle(rows)
        take = max(1, math.ceil(len(rows) * (percent / 100.0)))
        subset.extend(rows[:take])

    rng.shuffle(subset)
    return subset


def _prepare_preview_rows(samples: List[Dict], limit: int = 5) -> List[Dict]:
    previews = []
    for sample in samples[:limit]:
        previews.append(
            {
                "id": sample.get("id"),
                "label": sample.get("label"),
                "attack_type": sample.get("attack_type"),
                "prompt": _build_user_prompt(sample),
                "target": _build_target_json(sample),
            }
        )
    return previews


class Layer3SFTDataset(Dataset):
    def __init__(self, samples: List[Dict], tokenizer, max_length: int):
        self.rows = []
        self.skipped = 0

        for sample in samples:
            prompt_messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(sample)},
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": _build_target_json(sample)}
            ]

            prompt_text = _apply_chat_template(tokenizer, prompt_messages, add_generation_prompt=True)
            full_text = _apply_chat_template(tokenizer, full_messages, add_generation_prompt=False)

            prompt_tokens = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                padding=False,
            )
            full_tokens = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )

            input_ids = full_tokens["input_ids"]
            attention_mask = full_tokens["attention_mask"]
            labels = list(input_ids)

            prompt_len = min(len(prompt_tokens["input_ids"]), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
            labels = [
                label if mask == 1 else -100
                for label, mask in zip(labels, attention_mask)
            ]

            if all(label == -100 for label in labels):
                self.skipped += 1
                continue

            self.rows.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels,
                }
            )

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        item = self.rows[index]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


@dataclass
class RunSummary:
    output_dir: str
    hf_export_dir: str
    ollama_modelfile: str
    train_samples: int
    val_samples: int
    train_percent: int
    tuning_mode: str


def _resolve_base_model_name(base_model_name: str = None) -> str:
    if base_model_name:
        return base_model_name
    return os.getenv("LAYER3_SFT_BASE_MODEL", Layer3Llama31SFTConfig.MODEL_NAME)


def _load_tokenizer(model_name: str):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except OSError as exc:
        raise RuntimeError(
            "Failed to load the Hugging Face tokenizer for the Layer 3 base model.\n"
            f"Base model: {model_name}\n"
            "If this is a gated model, run `hf auth login` and make sure your account "
            "has been approved for that repository. You can also override the base "
            "model with the LAYER3_SFT_BASE_MODEL environment variable."
        ) from exc
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _pick_dtype():
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16, False, True
        return torch.float16, True, False
    return torch.float32, False, False


def _build_quantization_config(config: Layer3Llama31SFTConfig):
    if not config.USE_QLORA:
        return None
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "QLoRA was requested but `peft` is not installed in the current environment.\n"
            "Install it with: pip install peft"
        )
    if not BNB_AVAILABLE:
        raise RuntimeError(
            "QLoRA was requested but `bitsandbytes` is not installed in the current environment.\n"
            "Install it with: pip install bitsandbytes"
        )

    compute_dtype, _, _ = _pick_dtype()
    return BitsAndBytesConfig(
        load_in_4bit=config.LOAD_IN_4BIT,
        bnb_4bit_quant_type=config.BNB_4BIT_QUANT_TYPE,
        bnb_4bit_use_double_quant=config.BNB_4BIT_USE_DOUBLE_QUANT,
        bnb_4bit_compute_dtype=compute_dtype,
    )


def _load_model(model_name: str, config: Layer3Llama31SFTConfig):
    dtype, use_fp16, use_bf16 = _pick_dtype()
    quantization_config = _build_quantization_config(config)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=True,
            quantization_config=quantization_config,
            device_map="auto" if quantization_config is not None else None,
        )
    except OSError as exc:
        raise RuntimeError(
            "Failed to load the Hugging Face model weights for the Layer 3 base model.\n"
            f"Base model: {model_name}\n"
            "If this is a gated model, run `hf auth login` and make sure your account "
            "has been approved for that repository. You can also override the base "
            "model with the LAYER3_SFT_BASE_MODEL environment variable."
        ) from exc
    except Exception as exc:
        if quantization_config is not None:
            raise RuntimeError(
                "Failed to load the Layer 3 base model with QLoRA 4-bit settings.\n"
                "Please make sure CUDA, bitsandbytes, and the current PyTorch build are compatible."
            ) from exc
        raise

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if config.USE_GRADIENT_CHECKPOINTING:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    return model, use_fp16, use_bf16, quantization_config is not None


def _attach_lora_if_available(model, config: Layer3Llama31SFTConfig):
    if not PEFT_AVAILABLE:
        raise RuntimeError(
            "This Layer 3 fine-tuning pipeline now requires `peft` because it uses LoRA/QLoRA.\n"
            "Install it with: pip install peft"
        )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        lora_dropout=config.LORA_DROPOUT,
        target_modules=config.LORA_TARGET_MODULES,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    return model, "lora"


def _write_json(path: Path, payload: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _write_jsonl(path: Path, rows: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _export_ollama_scaffold(hf_export_dir: Path, run_dir: Path, config: Layer3Llama31SFTConfig):
    model_name = f"agent-defense-llama31-layer3-pct{run_dir.name.split('pct')[-1]}"
    stop_lines = "\n".join(
        f'PARAMETER stop "{token}"' for token in config.OLLAMA_STOP_TOKENS
    )
    modelfile = f"""FROM {hf_export_dir.resolve()}

PARAMETER temperature 0.1
{stop_lines}
"""
    modelfile_path = run_dir / "Modelfile"
    instructions_path = run_dir / "ollama_build_instructions.txt"

    modelfile_path.write_text(modelfile, encoding="utf-8")
    instructions_path.write_text(
        "\n".join(
            [
                "After training completes, you can register the exported model in Ollama with:",
                f"ollama create {model_name} -f {modelfile_path}",
                "",
                "This Modelfile intentionally does not set a SYSTEM prompt because Layer3Detector",
                "already sends the security-auditor system prompt at runtime.",
                "",
                "Then update Layer 3 config to use that Ollama model name if desired.",
            ]
        ) + "\n",
        encoding="utf-8",
    )
    return modelfile_path, instructions_path


def _merge_adapter_for_export(base_model_name: str, adapter_dir: Path, hf_export_dir: Path):
    dtype, _, _ = _pick_dtype()
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        dtype=dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    merged = PeftModel.from_pretrained(base_model, str(adapter_dir))
    merged = merged.merge_and_unload()
    hf_export_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(hf_export_dir, safe_serialization=True)
    return merged


def run(train_percent: int = 100, base_model_name: str = None) -> RunSummary:
    if not 1 <= int(train_percent) <= 100:
        raise ValueError("train_percent must be between 1 and 100")

    config = Layer3Llama31SFTConfig()
    base_model_name = _resolve_base_model_name(base_model_name)

    train_rows = load_jsonl(config.DATA_DIR / config.TRAIN_FILE)
    val_rows = load_jsonl(config.DATA_DIR / config.VAL_FILE)

    if not train_rows:
        raise ValueError(f"Empty training file: {config.DATA_DIR / config.TRAIN_FILE}")
    if not val_rows:
        raise ValueError(f"Empty validation file: {config.DATA_DIR / config.VAL_FILE}")

    selected_train = _stratified_subset(train_rows, int(train_percent), config.SEED)
    run_dir = config.OUTPUT_ROOT / f"llama31_8b_layer3_sft_pct{int(train_percent):03d}"
    hf_export_dir = run_dir / "hf_model"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nBase model          : {base_model_name}")
    print(f"Train percent       : {train_percent}%")
    print(f"Selected train rows : {len(selected_train)} / {len(train_rows)}")
    print(f"Validation rows     : {len(val_rows)}")
    print(f"Output directory    : {run_dir}")
    print(f"PEFT available      : {PEFT_AVAILABLE}")
    print(f"bitsandbytes avail. : {BNB_AVAILABLE}")

    tokenizer = _load_tokenizer(base_model_name)
    train_dataset = Layer3SFTDataset(selected_train, tokenizer, config.MAX_LENGTH)
    val_dataset = Layer3SFTDataset(val_rows, tokenizer, config.MAX_LENGTH)

    print(f"Usable train rows   : {len(train_dataset)} (skipped {train_dataset.skipped})")
    print(f"Usable val rows     : {len(val_dataset)} (skipped {val_dataset.skipped})")

    _write_jsonl(run_dir / "train_preview.jsonl", _prepare_preview_rows(selected_train))
    _write_jsonl(run_dir / "val_preview.jsonl", _prepare_preview_rows(val_rows))

    model, use_fp16, use_bf16, used_qlora = _load_model(base_model_name, config)
    model, tuning_mode = _attach_lora_if_available(model, config)
    print(f"Training mode       : {'QLoRA 4-bit + LoRA' if used_qlora else 'LoRA'}")

    effective_eval_steps = max(1, min(config.EVAL_STEPS, max(1, len(train_dataset) // max(1, config.BATCH_SIZE))))
    learning_rate = config.LR_LORA

    training_args = TrainingArguments(
        output_dir=str(run_dir / "trainer_output"),
        num_train_epochs=config.EPOCHS,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.EVAL_BATCH_SIZE,
        gradient_accumulation_steps=config.GRAD_ACCUM,
        learning_rate=learning_rate,
        weight_decay=config.WEIGHT_DECAY,
        warmup_ratio=config.WARMUP_RATIO,
        max_grad_norm=config.MAX_GRAD_NORM,
        eval_strategy="steps",
        eval_steps=effective_eval_steps,
        save_strategy="steps",
        save_steps=effective_eval_steps,
        save_total_limit=config.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=config.LOGGING_STEPS,
        report_to=config.REPORT_TO,
        seed=config.SEED,
        data_seed=config.SEED,
        dataloader_num_workers=config.DATALOADER_NUM_WORKERS,
        remove_unused_columns=False,
        fp16=use_fp16,
        bf16=use_bf16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=default_data_collator,
    )

    trainer.train()
    eval_metrics = trainer.evaluate()

    adapter_dir = run_dir / "adapter_model"
    trainer.model.save_pretrained(adapter_dir)
    hf_export_dir.mkdir(parents=True, exist_ok=True)
    _merge_adapter_for_export(base_model_name, adapter_dir, hf_export_dir)
    tokenizer.save_pretrained(hf_export_dir)

    modelfile_path, instructions_path = _export_ollama_scaffold(hf_export_dir, run_dir, config)

    training_config = {
        "base_model_name": base_model_name,
        "ollama_model_name": config.OLLAMA_MODEL_NAME,
        "train_percent": int(train_percent),
        "train_rows_total": len(train_rows),
        "train_rows_selected": len(selected_train),
        "train_rows_used": len(train_dataset),
        "train_rows_skipped": train_dataset.skipped,
        "val_rows_total": len(val_rows),
        "val_rows_used": len(val_dataset),
        "val_rows_skipped": val_dataset.skipped,
        "max_length": config.MAX_LENGTH,
        "max_new_tokens": config.MAX_NEW_TOKENS,
        "epochs": config.EPOCHS,
        "learning_rate": learning_rate,
        "batch_size": config.BATCH_SIZE,
        "eval_batch_size": config.EVAL_BATCH_SIZE,
        "gradient_accumulation": config.GRAD_ACCUM,
        "weight_decay": config.WEIGHT_DECAY,
        "warmup_ratio": config.WARMUP_RATIO,
        "tuning_mode": tuning_mode,
        "use_qlora": used_qlora,
        "peft_available": PEFT_AVAILABLE,
        "bitsandbytes_available": BNB_AVAILABLE,
        "adapter_dir": str(adapter_dir),
        "output_dir": str(run_dir),
        "hf_export_dir": str(hf_export_dir),
        "ollama_modelfile": str(modelfile_path),
        "ollama_build_instructions": str(instructions_path),
        "eval_metrics": eval_metrics,
    }
    _write_json(run_dir / "training_config.json", training_config)

    print("\nFine-tuning completed.")
    print(f"HF export           : {hf_export_dir}")
    print(f"Ollama Modelfile    : {modelfile_path}")
    print(f"Build instructions  : {instructions_path}")

    return RunSummary(
        output_dir=str(run_dir),
        hf_export_dir=str(hf_export_dir),
        ollama_modelfile=str(modelfile_path),
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        train_percent=int(train_percent),
        tuning_mode=tuning_mode,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the local Layer 3 Llama model.")
    parser.add_argument("--train_percent", type=int, default=100, help="Percentage of training data to use (1-100).")
    parser.add_argument("--base_model_name", type=str, default=None, help="Optional Hugging Face base model override.")
    args = parser.parse_args()
    run(train_percent=args.train_percent, base_model_name=args.base_model_name)
