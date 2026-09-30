# ============================================================================
# FILE: models/baseline/raguard.py
# RAG-channel baseline adapted from RAGuard for offline trace evaluation.
#
# Purpose:
# - Provides the retrieval-only baseline used for Issue 3 comparisons.
#
# Workflow:
# - Extracts documents from the retrieve step.
# - Computes retrieval-focused signals such as perplexity and semantic
#   consistency.
# - Produces prediction and metric artifacts for downstream evaluation.
#
# Use this file when:
# - You want to inspect or rerun the RAG-only baseline.
# - You need to compare retrieval-focused defense against the 3-layer system.
# ============================================================================
# raguard_detector.py

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from tqdm.auto import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from utils.utils import load_jsonl, save_jsonl
from sentence_transformers import SentenceTransformer


class RAGuardDetector:
    """
    Offline trace-level adaptation of RAGuard for raw agent trace JSONL.

    Expected sample format:
    {
        "id": "...",
        "source": "rag",
        "attack_type": "rag",
        "prompt": "...",
        "reasoning_steps": [
            {"node": "retrieve", "type": "agent", "content": "... [Doc 1] ... [Doc 2] ..."},
            {"node": "agent", ...},
            {"node": "tools", ...}
        ],
        "label": 0 or 1
    }

    Core design:
    - ONLY inspect RAG channel
    - ONLY read:
        * sample["prompt"]
        * retrieve step content
    - DO NOT inspect:
        * agent outputs
        * tool outputs
        * other channels
    """

    DOC_PATTERN = re.compile(r"\[Doc\s*\d+\](.*?)(?=\[Doc\s*\d+\]|\Z)", re.S)

    def __init__(
        self,
        alpha: float = 0.025,
        gpt2_model_name: str = "gpt2",
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_ppl_tokens: int = 512,
        min_words_per_doc: int = 20,
        use_ts: bool = True,
        device: Optional[str] = None,
        verbose: bool = True,
    ):
        self.alpha = alpha
        self.gpt2_model_name = gpt2_model_name
        self.embedding_model_name = embedding_model_name
        self.max_ppl_tokens = max_ppl_tokens
        self.min_words_per_doc = min_words_per_doc
        self.use_ts = use_ts
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        if self.verbose:
            print(f"[RAGuardDetector] Using device: {self.device}")
            print(f"[RAGuardDetector] Loading GPT-2: {self.gpt2_model_name}")

        self.tokenizer = GPT2TokenizerFast.from_pretrained(self.gpt2_model_name)
        self.lm = GPT2LMHeadModel.from_pretrained(self.gpt2_model_name).to(self.device)
        self.lm.eval()

        self.embedder = None
        if self.use_ts:
            if self.verbose:
                print(f"[RAGuardDetector] Loading embedding model: {self.embedding_model_name}")
            self.embedder = SentenceTransformer(self.embedding_model_name, device=self.device)

        # thresholds after fitting
        self.thresholds: Optional[Dict[str, float]] = None

    # ---------------------------------------------------------------------
    # Parsing
    # ---------------------------------------------------------------------
    def extract_query(self, sample: Dict[str, Any]) -> str:
        return (sample.get("prompt") or "").strip()

    def extract_retrieve_text(self, sample: Dict[str, Any]) -> str:
        reasoning_steps = sample.get("reasoning_steps", [])
        for step in reasoning_steps:
            if step.get("node") == "retrieve":
                return (step.get("content") or "").strip()
        return ""

    def extract_docs(self, retrieve_text: str) -> List[str]:
        docs = self.DOC_PATTERN.findall(retrieve_text)
        return [doc.strip() for doc in docs if doc and doc.strip()]

    def get_docs_from_sample(self, sample: Dict[str, Any]) -> List[str]:
        retrieve_text = self.extract_retrieve_text(sample)
        return self.extract_docs(retrieve_text)
    
    @staticmethod
    def compute_confusion(y_true, y_pred):

        TP = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
        FN = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
        TN = sum((t == 0 and p == 0) for t, p in zip(y_true, y_pred))
        FP = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))

        return TP, FN, TN, FP

    def compute_per_attack_type(self, dataset, predictions):

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

                y_true.append(sample["label"])
                y_pred.append(pred["pred"])

            TP, FN, TN, FP = self.compute_confusion(y_true, y_pred)

            total_attack = TP + FN
            total_benign = TN + FP

            ASR = FN / total_attack if total_attack > 0 else 0
            TSR = TP / total_attack if total_attack > 0 else 0
            FPR = FP / total_benign if total_benign > 0 else 0

            results[atk] = {
                "TP": TP,
                "FN": FN,
                "TN": TN,
                "FP": FP,
                "ASR": ASR,
                "TSR": TSR,
                "FPR": FPR
            }

        return results
    
    def generate_metrics(self, dataset, predictions):

        from sklearn.metrics import classification_report

        y_true = [p["label"] for p in predictions]
        y_pred = [p["pred"] for p in predictions]

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

    def save_metrics(self, metrics, path):

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

    # ---------------------------------------------------------------------
    # Scoring
    # ---------------------------------------------------------------------
    def split_doc(self, text: str) -> Tuple[str, str]:
        words = text.split()

        if len(words) < self.min_words_per_doc:
            return text, text

        mid = len(words) // 2
        pre = " ".join(words[:mid]).strip()
        post = " ".join(words[mid:]).strip()

        if not pre:
            pre = text
        if not post:
            post = text

        return pre, post

    def compute_perplexity(self, text: str) -> float:
        text = (text or "").strip()
        if not text:
            return float("inf")

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_ppl_tokens,
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids,
            )
            loss = outputs.loss
            ppl = torch.exp(loss).item()

        return float(ppl)

    def compute_similarity(self, query: str, doc: str) -> float:
        if not self.use_ts:
            return float("nan")

        query = (query or "").strip()
        doc = (doc or "").strip()

        if not query or not doc:
            return float("nan")

        q_emb = self.embedder.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        d_emb = self.embedder.encode(doc, convert_to_tensor=True, normalize_embeddings=True)

        sim = F.cosine_similarity(q_emb.unsqueeze(0), d_emb.unsqueeze(0)).item()
        return float(sim)

    def score_doc(self, query: str, doc: str) -> Dict[str, float]:
        pre, post = self.split_doc(doc)

        ppl_pre = self.compute_perplexity(pre)
        ppl_post = self.compute_perplexity(post)

        pd_signed = float(ppl_pre - ppl_post)
        pd_abs = float(abs(pd_signed))
        pm = float(max(ppl_pre, ppl_post))
        ts = float(self.compute_similarity(query, doc)) if self.use_ts else float("nan")

        return {
            "ppl_pre": float(ppl_pre),
            "ppl_post": float(ppl_post),
            "pd_signed": pd_signed,
            "pd_abs": pd_abs,
            "pm": pm,
            "ts": ts,
        }

    # ---------------------------------------------------------------------
    # Fitting thresholds
    # ---------------------------------------------------------------------
    def fit(self, dataset: List[Dict[str, Any]], benign_label: int = 0) -> Dict[str, float]:
        pd_vals = []
        pm_vals = []
        ts_vals = []

        iterator = tqdm(dataset, desc="Fitting RAGuard", disable=not self.verbose)

        for sample in iterator:
            if int(sample.get("label", -1)) != benign_label:
                continue

            query = self.extract_query(sample)
            docs = self.get_docs_from_sample(sample)

            for doc in docs:
                if len(doc.split()) < self.min_words_per_doc:
                    continue

                try:
                    scores = self.score_doc(query, doc)
                    pd_vals.append(scores["pd_signed"])
                    pm_vals.append(scores["pm"])

                    if self.use_ts and not np.isnan(scores["ts"]):
                        ts_vals.append(scores["ts"])
                except Exception as e:
                    if self.verbose:
                        print(f"[WARN] Skipping doc during fit due to error: {e}")
                    continue

        if len(pd_vals) == 0 or len(pm_vals) == 0:
            raise ValueError("No valid benign retrieval documents were found to fit thresholds.")

        thresholds = {
            "pd_low": float(np.percentile(pd_vals, self.alpha * 100)),
            "pd_high": float(np.percentile(pd_vals, (1 - self.alpha) * 100)),
            "pm_high": float(np.percentile(pm_vals, (1 - self.alpha) * 100)),
        }

        if self.use_ts:
            if len(ts_vals) == 0:
                raise ValueError("TS is enabled but no valid similarity scores were collected.")
            thresholds["ts_high"] = float(np.percentile(ts_vals, (1 - self.alpha) * 100))

        self.thresholds = thresholds

        if self.verbose:
            print("\n[RAGuardDetector] Learned thresholds:")
            for k, v in thresholds.items():
                print(f"  {k}: {v:.6f}")

        return thresholds

    def fit_from_file(self, train_path: str, benign_label: int = 0) -> Dict[str, float]:
        dataset = load_jsonl(train_path)
        return self.fit(dataset, benign_label=benign_label)

    # ---------------------------------------------------------------------
    # Threshold persistence
    # ---------------------------------------------------------------------
    def save_thresholds(self, path: str) -> None:
        if self.thresholds is None:
            raise RuntimeError("Thresholds are not fitted yet.")

        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)

        with open(path_obj, "w", encoding="utf-8") as f:
            json.dump(self.thresholds, f, ensure_ascii=False, indent=2)

        if self.verbose:
            print(f"[RAGuardDetector] Thresholds saved to: {path_obj}")

    def load_thresholds(self, path: str) -> Dict[str, float]:
        with open(path, "r", encoding="utf-8") as f:
            self.thresholds = json.load(f)

        if self.verbose:
            print(f"[RAGuardDetector] Thresholds loaded from: {path}")
            for k, v in self.thresholds.items():
                print(f"  {k}: {v:.6f}")

        return self.thresholds

    # ---------------------------------------------------------------------
    # Prediction
    # ---------------------------------------------------------------------
    def is_doc_suspicious(self, scores: Dict[str, float]) -> Tuple[bool, List[str]]:
        if self.thresholds is None:
            raise RuntimeError("Thresholds are not available. Call fit() or load_thresholds() first.")

        reasons = []

        if scores["pd_signed"] <= self.thresholds["pd_low"]:
            reasons.append("pd_low_tail")

        if scores["pd_signed"] >= self.thresholds["pd_high"]:
            reasons.append("pd_high_tail")

        if scores["pm"] >= self.thresholds["pm_high"]:
            reasons.append("pm_high")

        if self.use_ts and "ts_high" in self.thresholds and not np.isnan(scores["ts"]):
            if scores["ts"] >= self.thresholds["ts_high"]:
                reasons.append("ts_high")

        return len(reasons) > 0, reasons

    def predict_sample(self, sample: Dict[str, Any], return_details: bool = True) -> Dict[str, Any]:
        if self.thresholds is None:
            raise RuntimeError("Thresholds are not available. Call fit() or load_thresholds() first.")

        query = self.extract_query(sample)
        docs = self.get_docs_from_sample(sample)

        pred = 0
        doc_results = []

        for idx, doc in enumerate(docs, start=1):
            if len(doc.split()) < self.min_words_per_doc:
                doc_results.append({
                    "doc_index": idx,
                    "pred": 0,
                    "skipped": True,
                    "reason": "too_short",
                })
                continue

            try:
                scores = self.score_doc(query, doc)
                suspicious, reasons = self.is_doc_suspicious(scores)

                if suspicious:
                    pred = 1

                doc_results.append({
                    "doc_index": idx,
                    "pred": int(suspicious),
                    "reasons": reasons,
                    **scores,
                })

            except Exception as e:
                doc_results.append({
                    "doc_index": idx,
                    "pred": 0,
                    "skipped": True,
                    "reason": f"error: {str(e)}",
                })

        output = sample.copy()

        output.update({
            "pred": int(pred),
            "raguard_num_docs": len(docs),
        })

        if return_details:
            output["raguard_doc_results"] = doc_results

        return output

    def predict(self, dataset: List[Dict[str, Any]], return_details: bool = True) -> List[Dict[str, Any]]:
        results = []
        iterator = tqdm(dataset, desc="Predicting with RAGuard", disable=not self.verbose)

        for sample in iterator:
            results.append(self.predict_sample(sample, return_details=return_details))

        return results

    def predict_from_file(self, path: str, return_details: bool = True) -> List[Dict[str, Any]]:
        dataset = load_jsonl(path)
        return self.predict(dataset, return_details=return_details)

    # ---------------------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------------------
    def evaluate(self, dataset: List[Dict[str, Any]], return_details: bool = True):
        predictions = self.predict(dataset, return_details=return_details)

        y_true = [p["label"] for p in predictions]
        y_pred = [p["pred"] for p in predictions]

        report_text = classification_report(y_true, y_pred, digits=4, zero_division=0)
        report_dict = classification_report(y_true, y_pred, digits=4, zero_division=0, output_dict=True)

        if self.verbose:
            print("\n[RAGuardDetector] Classification Report")
            print(report_text)

        return report_dict, predictions

    def evaluate_from_file(self, test_path: str, return_details: bool = True):
        dataset = load_jsonl(test_path)
        return self.evaluate(dataset, return_details=return_details)
    

def run_raguard():

    TRAIN_PATH = Path("data/splits/train.jsonl")
    TEST_PATH = Path("data/splits/test.jsonl")

    THRESHOLD_PATH = Path("models/baseline/raguard_thresholds.json")
    PREDICTION_PATH = Path("models/baseline/raguard_predictions.jsonl")
    METRIC_PATH = Path("models/baseline/raguard_metrics.json")

    detector = RAGuardDetector(
        alpha=0.025,
        max_ppl_tokens=512,
        min_words_per_doc=20,
        use_ts=True,
        verbose=True,
    )

    # fit threshold
    detector.fit_from_file(TRAIN_PATH)

    detector.save_thresholds(THRESHOLD_PATH)

    # load dataset
    dataset = load_jsonl(TEST_PATH)

    import time
    start = time.time()
    # evaluate
    report, predictions = detector.evaluate(dataset)
    end = time.time()
    total_time = end - start
    save_jsonl(predictions, PREDICTION_PATH)

    # generate metrics
    metrics = detector.generate_metrics(dataset, predictions)
    metrics["time"] = total_time
    detector.save_metrics(metrics, METRIC_PATH)

    print("\nDone.")
    print(f"Thresholds saved to: {THRESHOLD_PATH}")
    print(f"Predictions saved to: {PREDICTION_PATH}")
    print(f"Metrics saved to: {METRIC_PATH}")
