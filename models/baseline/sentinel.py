# ============================================================================
# FILE: models/baseline/sentinel.py
# Prompt-channel baseline using the Prompt Injection Sentinel model.
#
# Purpose:
# - Serves as the prompt-only baseline for Issue 3 comparisons.
#
# Workflow:
# - Reads only the original prompt channel from each trace.
# - Runs the Sentinel classifier over that prompt.
# - Saves predictions, metrics, and threshold/config metadata.
#
# Use this file when:
# - You want to inspect or rerun the prompt-only baseline.
# - You need to compare the layered defense against a prompt-channel detector.
# ============================================================================
# prompt_channel_sentinel_detector.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from utils.utils import load_jsonl, save_jsonl


class PromptChannelSentinelDetector:
    """
    Offline trace-level prompt-channel detector using the SOTA Prompt Injection Sentinel model.

    Expected sample format:
    {
        "id": "...",
        "source": "prompt" / "rag" / "tool" / "correlated" / ...,
        "attack_type": "...",
        "prompt": "...",
        "reasoning_steps": [...],
        "label": 0 or 1
    }

    Core design:
    - ONLY inspect PROMPT channel
    - ONLY read:
        * sample["prompt"]
    - DO NOT inspect:
        * retrieve content
        * tool outputs
        * agent outputs
        * other channels

    Prediction semantics:
    - pred = 1 -> suspicious / malicious prompt detected
    - pred = 0 -> benign prompt
    """

    def __init__(
        self,
        model_name: str = "qualifire/prompt-injection-sentinel",
        max_length: int = 512,
        batch_size: int = 16,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        if self.verbose:
            print(f"[PromptChannelSentinelDetector] Using device: {self.device}")
            print(f"[PromptChannelSentinelDetector] Loading model: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name).to(self.device)
        self.model.eval()

        # kept for style consistency with other baselines
        self.thresholds: Dict[str, Any] = {
            "detector_name": "prompt_channel_sentinel",
            "defense_stage": "prompt_channel",
            "method": "PromptInjectionSentinel",
            "model_name": self.model_name,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "decision_rule": "argmax_over_logits",
        }

    # ---------------------------------------------------------------------
    # File utilities
    # ---------------------------------------------------------------------
    def save_metrics(self, metrics: Dict[str, Any], path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=4)

    def save_thresholds(self, path: str) -> None:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"[PromptChannelSentinelDetector] Config saved to: {path_obj}")

    def load_thresholds(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

        if self.verbose:
            print(f"[PromptChannelSentinelDetector] Config loaded from: {path}")

        return self.thresholds

    # ---------------------------------------------------------------------
    # Prompt parsing
    # ---------------------------------------------------------------------
    def extract_prompt(self, sample: Dict[str, Any]) -> str:
        return (sample.get("prompt") or "").strip()

    # ---------------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------------
    @staticmethod
    def compute_confusion(y_true: List[int], y_pred: List[int]) -> Tuple[int, int, int, int]:
        TP = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
        FN = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
        TN = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
        FP = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
        return TP, FN, TN, FP

    def compute_per_attack_type(self, dataset: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        results = {}

        attack_types = set(
            sample.get("attack_type", "unknown")
            for sample in dataset
            if sample.get("label") == 1
        )

        for atk in attack_types:
            y_true = []
            y_pred = []

            for sample, pred in zip(dataset, predictions):
                if sample.get("attack_type") != atk:
                    continue

                y_true.append(int(sample["label"]))
                y_pred.append(int(pred["pred"]))

            TP, FN, TN, FP = self.compute_confusion(y_true, y_pred)

            total_attack = TP + FN
            total_benign = TN + FP

            ASR = FN / total_attack if total_attack > 0 else 0.0
            TSR = TP / total_attack if total_attack > 0 else 0.0
            FPR = FP / total_benign if total_benign > 0 else 0.0

            results[atk] = {
                "TP": TP,
                "FN": FN,
                "TN": TN,
                "FP": FP,
                "ASR": ASR,
                "TSR": TSR,
                "FPR": FPR,
            }

        return results

    def generate_metrics(self, dataset: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        y_true = [int(p["label"]) for p in predictions]
        y_pred = [int(p["pred"]) for p in predictions]

        report = classification_report(
            y_true,
            y_pred,
            digits=4,
            zero_division=0,
            output_dict=True
        )

        TP, FN, TN, FP = self.compute_confusion(y_true, y_pred)

        metrics = {
            "overall": {
                "TP": TP,
                "FN": FN,
                "TN": TN,
                "FP": FP
            },
            "classification_report": report,
            "per_attack_type": self.compute_per_attack_type(dataset, predictions)
        }

        return metrics

    # ---------------------------------------------------------------------
    # Model inference
    # ---------------------------------------------------------------------
    def _infer_batch(self, texts: List[str]) -> Tuple[List[int], List[List[float]]]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        preds = torch.argmax(logits, dim=-1).detach().cpu().tolist()
        probs_list = probs.detach().cpu().tolist()

        return preds, probs_list

    def predict_sample(self, sample: Dict[str, Any], return_details: bool = True) -> Dict[str, Any]:
        prompt_text = self.extract_prompt(sample)

        if not prompt_text:
            output = sample.copy()
            output.update({
                "pred": 0,
                "sentinel_prompt_text": "",
            })
            if return_details:
                output["sentinel_scores"] = {
                    "label_pred": 0,
                    "probs": [],
                    "note": "empty_prompt_default_benign",
                }
            return output

        preds, probs_list = self._infer_batch([prompt_text])
        pred = int(preds[0])
        probs = probs_list[0]

        output = sample.copy()
        output.update({
            "pred": pred,
            "sentinel_prompt_text": prompt_text,
        })

        if return_details:
            output["sentinel_scores"] = {
                "label_pred": pred,
                "probs": probs,
            }

        return output

    def predict(self, dataset: List[Dict[str, Any]], return_details: bool = True) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []

        prompts = [self.extract_prompt(sample) for sample in dataset]
        iterator = tqdm(
            range(0, len(dataset), self.batch_size),
            desc="Predicting with PromptChannelSentinel",
            disable=not self.verbose,
        )

        for start in iterator:
            end = min(start + self.batch_size, len(dataset))
            batch_samples = dataset[start:end]
            batch_prompts = prompts[start:end]

            non_empty_indices = [i for i, p in enumerate(batch_prompts) if p.strip()]
            batch_preds = [0] * len(batch_samples)
            batch_probs: List[List[float]] = [[] for _ in range(len(batch_samples))]

            if non_empty_indices:
                texts = [batch_prompts[i] for i in non_empty_indices]
                preds, probs_list = self._infer_batch(texts)

                for local_idx, pred, probs in zip(non_empty_indices, preds, probs_list):
                    batch_preds[local_idx] = int(pred)
                    batch_probs[local_idx] = probs

            for sample, prompt_text, pred, probs in zip(batch_samples, batch_prompts, batch_preds, batch_probs):
                output = sample.copy()
                output.update({
                    "pred": int(pred),
                    "sentinel_prompt_text": prompt_text,
                })

                if return_details:
                    output["sentinel_scores"] = {
                        "label_pred": int(pred),
                        "probs": probs,
                    }

                results.append(output)

        return results

    def predict_from_file(self, path: str, return_details: bool = True) -> List[Dict[str, Any]]:
        dataset = load_jsonl(path)
        return self.predict(dataset, return_details=return_details)

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def evaluate(self, dataset: List[Dict[str, Any]], return_details: bool = True):
        predictions = self.predict(dataset, return_details=return_details)

        y_true = [int(p["label"]) for p in predictions]
        y_pred = [int(p["pred"]) for p in predictions]

        report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
        report_dict = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)

        if self.verbose:
            print("\n[PromptChannelSentinelDetector] Classification Report")
            print(report_text)

        return report_dict, predictions

    def evaluate_from_file(self, test_path: str, return_details: bool = True):
        dataset = load_jsonl(test_path)
        return self.evaluate(dataset, return_details=return_details)


def run_prompt_channel_sentinel():
    TEST_PATH = Path("data/splits/test.jsonl")

    THRESHOLD_PATH = Path("models/baseline/sentinel_thresholds.json")
    PREDICTION_PATH = Path("models/baseline/sentinel_predictions.jsonl")
    METRIC_PATH = Path("models/baseline/sentinel_metrics.json")

    detector = PromptChannelSentinelDetector(
        model_name="qualifire/prompt-injection-sentinel",
        max_length=2048,
        batch_size=8,
        verbose=True,
    )

    detector.save_thresholds(THRESHOLD_PATH)

    dataset = load_jsonl(TEST_PATH)

    import time
    start = time.time()

    report, predictions = detector.evaluate(dataset)


    end = time.time()
    total_time = end - start

    save_jsonl(predictions, PREDICTION_PATH)

    metrics = detector.generate_metrics(dataset, predictions)
    metrics["time"] = total_time
    detector.save_metrics(metrics, METRIC_PATH)

    print("\nDone.")
    print(f"Config saved to: {THRESHOLD_PATH}")
    print(f"Predictions saved to: {PREDICTION_PATH}")
    print(f"Metrics saved to: {METRIC_PATH}")


if __name__ == "__main__":
    run_prompt_channel_sentinel()
