# ============================================================================
# FILE: models/detectors/pipeline.py
# Multi-layer routing pipeline for all individual and chained defense modes.
#
# Purpose:
# - Implements the execution logic that connects Layer 1, Layer 2, and Layer 3
#   under different experimental routing strategies.
#
# Workflow:
# - Loads mode definitions from config.
# - Instantiates only the layers required by the selected mode.
# - Routes samples through individual or chained execution paths.
# - Produces predictions, metrics, and flow-analysis summaries.
#
# Supported modes:
# - Individual: 1 (L1), 2 (L2), 3 (L3)
# - Chained: 10 (L1->L2), 11 (L1->L2->L3), 12 (L2->L1->L3),
#   13 (L1->L3), 14 (L2->L3), 15 (L2->L1)
#
# Use this file when:
# - You need to understand how samples move through the 3-layer system.
# - You want to modify the experiment routing logic.
# ============================================================================
from __future__ import annotations

import time
import yaml
import numpy as np
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

# If your project structure already makes `utils` importable, you can remove sys.path hacks.
from utils.metrics import (
    compute_complete_metrics,
    compute_roc_curve_data,
    compute_precision_recall_curve_data,
)
from utils.flow_analyzer import analyze_flow, generate_flow_diagram_data

load_dotenv()


class DetectionPipeline:
    """
    Design-accurate routing:

    Layer 1 (AE anomaly):
      - pred=0: benign -> FORWARD (done)
      - pred=1: suspicious -> NEXT

    Layer 2 (Supervised classifier):
      - pred=1: malicious -> BLOCK (done)
      - pred=0: not malicious (unknown/benign-ish) -> NEXT

    Layer 3 (LLM):
      - pred=0: benign -> FORWARD
      - pred=1: malicious -> BLOCK

    Supported modes:
      1: L1 only (individual)
      2: L2 only (individual)
      3: L3 only (individual)
     10: L1 -> L2
     11: L1 -> L2 -> L3
     12: L2 -> L1 -> L3
     13: L1 -> L3
     14: L2 -> L3
     15: L2 -> L1
     16: L1 -> L2 -> L3V4
    """
    # L2_TAU_LOW = 0.2
    # L2_TAU_HIGH = 0.8

    # -------------------------
    # Init
    # -------------------------
    def __init__(
        self,
        config_path: str = "config/detection_config.yaml",
        test_mode: Optional[int] = None,
        layer3_variant: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        self.config = self._load_yaml(self.config_path)

        if test_mode is not None:
            self.config["test_mode"] = int(test_mode)

        self.device: str = self._normalize_device(self.config.get("device", "cuda"))
        self.layers_cfg: Dict[str, Dict[str, Any]] = self.config.get("layers", {})
        self.layer3_variant: Optional[str] = self._normalize_layer3_variant(
            layer3_variant
            or os.getenv("AGENT_DEFENSE_LAYER3_VARIANT")
            or self.config.get("layer3_variant")
        )
        self.layer_aliases: Dict[str, str] = {}
        self.enabled_layers: Dict[str, Dict[str, Any]] = {
            name: cfg for name, cfg in self.layers_cfg.items() if cfg.get("enabled", True)
        }

        self._load_test_mode()
        self._load_layers()
        self._print_init()

    # -------------------------
    # YAML utils
    # -------------------------
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"YAML file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return self._expand_env_vars(yaml.safe_load(f) or {})

    @classmethod
    def _expand_env_vars(cls, value: Any) -> Any:
        if isinstance(value, str):
            return os.path.expandvars(value)
        if isinstance(value, dict):
            return {key: cls._expand_env_vars(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._expand_env_vars(item) for item in value]
        return value

    @staticmethod
    def _normalize_device(value: Any) -> str:
        device = str(value or "cuda").strip().lower()
        if device in {"cpu", "none", "false", "-1"}:
            return "cpu"
        if device.startswith("cuda"):
            return device
        if device.startswith("${"):
            return "cuda"
        first = device.split(",", 1)[0].strip()
        return f"cuda:{first}" if first else "cuda"

    def _load_test_mode(self) -> None:
        test_modes_path = Path(self.config.get("test_modes_file", "config/test_modes.yaml"))
        modes_cfg = self._load_yaml(test_modes_path)

        self.all_test_modes: Dict[int, Dict[str, Any]] = modes_cfg.get("test_modes", {})
        default_mode = int(modes_cfg.get("default_mode", 10))
        self.current_mode_id: int = int(self.config.get("test_mode", default_mode))

        if self.current_mode_id not in self.all_test_modes:
            raise ValueError(f"Test mode {self.current_mode_id} not found in {test_modes_path}")

        self.current_mode: Dict[str, Any] = self.all_test_modes[self.current_mode_id]

        # Hard restrict to your desired set
        allowed = {1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44}
        if self.current_mode_id not in allowed:
            raise ValueError(f"Only modes {sorted(allowed)} are supported, got {self.current_mode_id}")

        if not self.current_mode.get("enabled", True):
            # You may want to allow running disabled modes anyway; here we warn only.
            print(f"⚠️  Warning: test_mode {self.current_mode_id} is marked enabled:false but will run anyway.")

    @staticmethod
    def _normalize_layer3_variant(variant: Optional[str]) -> Optional[str]:
        if variant is None:
            return None

        value = str(variant).strip().lower()
        if value in {"", "default", "none", "current", "layer3"}:
            return None
        if value in {"o2", "layer3o2", "single", "channel", "channel-audit", "channelaudit"}:
            return "layer3o2"
        if value in {"1", "v1", "orig", "original", "legacy", "layer3orig", "layer3original"}:
            return "layer3orig"
        if value.startswith("layer3"):
            return value
        raise ValueError(
            "Invalid layer3_variant. Use default, orig, o2, or a configured layer key like layer3o2."
        )

    def _resolve_layer_name(self, logical_name: str) -> str:
        if logical_name == "layer3" and self.layer3_variant is not None:
            return self.layer3_variant
        return logical_name

    def _is_llm_layer(self, layer_name: str) -> bool:
        implementation_name = self.layer_aliases.get(layer_name, layer_name)
        return implementation_name.startswith("layer3")

    # -------------------------
    # Layer loading
    # -------------------------
    def _load_layers(self) -> None:
        from .layer1_detector import Layer1Detector
        from .layer2_detector import Layer2Detector
        from .layer3_detector import Layer3Detector
        from .layer3_detector_o2 import Layer3DetectorO2

        cls_map = {
            "layer1": Layer1Detector,
            "layer2": Layer2Detector,
            "layer3": Layer3DetectorO2,
            "layer3orig": Layer3Detector,
            "layer3o2": Layer3DetectorO2,
        }
        required_layers: List[str] = self.current_mode["layers"]

        self.layers: Dict[str, Any] = {}
        for name in required_layers:
            implementation_name = self._resolve_layer_name(name)
            if implementation_name not in self.enabled_layers:
                raise ValueError(
                    f"Layer '{implementation_name}' is required by mode {self.current_mode_id} "
                    f"but disabled in detection_config.yaml"
                )
            cls = cls_map.get(implementation_name)
            if cls is None:
                raise ValueError(f"Unknown layer implementation: {implementation_name}")

            params = (self.enabled_layers[implementation_name] or {}).get("params", {})
            self.layer_aliases[name] = implementation_name
            self.layers[name] = cls(device=self.device, **params)

    def _print_init(self) -> None:
        m = self.current_mode
        print("\n" + "=" * 80)
        print("🚀 DETECTION PIPELINE INITIALIZED (DESIGN-ACCURATE, SIMPLIFIED)")
        print("=" * 80)
        print(f"   Test Mode: {self.current_mode_id} - {m.get('name', 'unknown')}")
        print(f"   Layers   : {' → '.join(m.get('layers', []))}")
        if self.layer_aliases.get("layer3", "layer3") != "layer3":
            print(f"   Layer 3  : layer3 → {self.layer_aliases['layer3']}")
        print("=" * 80 + "\n")

    # -------------------------
    # Core runner helper
    # -------------------------
    def _run_layer(
        self,
        layer_name: str,
        sequences: List[str],
        **kwargs # paste the param of a specific layer
    ) -> Dict[str, Optional[np.ndarray]]:
        layer = self.layers[layer_name]
        out = layer.predict(sequences, **kwargs)
        preds = np.asarray(out["predictions"], dtype=int)
        probs = out.get("probabilities", None)
        if probs is None and "errors" in out:
            # Treat AE errors as anomaly score
            probs = out["errors"]

        prob_arr = np.asarray(probs, dtype=float) if probs is not None else None
        return {"predictions": preds, "probabilities": prob_arr, "raw": out}

    @staticmethod
    def _layer_confusion(labels: Optional[np.ndarray], preds: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Confusion matrix for the exact subset processed by one layer.

        For Layer 1 in cascaded modes, pred=1 means "suspicious/escalate", so:
        - TP: malicious samples escalated by Layer 1
        - FN: malicious samples not escalated by Layer 1
        - FP: benign samples escalated by Layer 1
        - TN: benign samples not escalated by Layer 1

        For Layer 2/3, pred=1 means "malicious/block" and pred=0 means
        "benign/pass to the next stage or final forward".
        """
        if labels is None:
            return None

        y_true = np.asarray(labels, dtype=int)
        y_pred = np.asarray(preds, dtype=int)
        if len(y_true) != len(y_pred):
            raise ValueError(
                f"Layer confusion length mismatch: labels={len(y_true)} preds={len(y_pred)}"
            )

        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())

        return {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "support": {
                "benign": int((y_true == 0).sum()),
                "malicious": int((y_true == 1).sum()),
                "total": int(len(y_true)),
            },
        }

    def _attach_layer_confusion(
        self,
        layer_stats: Dict[str, Dict[str, Any]],
        layer_name: str,
        labels: Optional[np.ndarray],
        preds: np.ndarray,
    ) -> None:
        confusion = self._layer_confusion(labels, preds)
        if confusion is not None:
            layer_stats[layer_name]["confusion_matrix"] = confusion

    def _finalize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        result["layer_implementations"] = dict(self.layer_aliases)
        if "layer3" in self.layer_aliases:
            result["selected_layer3_variant"] = self.layer_aliases["layer3"]
        return result
    
    # -------------------------
    # Public API
    # -------------------------
    def predict(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray] = None,
        return_details: bool = True,
    ) -> Dict[str, Any]:
        if self.current_mode_id in (1, 2, 3):
            return self._finalize_result(self._mode_individual(sequences, labels, return_details))

        if self.current_mode_id == 10:
            return self._finalize_result(self._mode10_l1_to_l2(sequences, labels, return_details))

        if self.current_mode_id == 11:
            return self._finalize_result(self._mode11_l1_to_l2_to_l3(sequences, labels, return_details))

        if self.current_mode_id == 12:
            return self._finalize_result(self._mode12_l2_to_l1_to_l3(sequences, labels, return_details))

        if self.current_mode_id == 13:
            return self._finalize_result(self._mode13_l1_to_l3(sequences, labels, return_details))

        if self.current_mode_id == 14:
            return self._finalize_result(self._mode14_l2_to_l3(sequences, labels, return_details))

        if self.current_mode_id == 15:
            return self._finalize_result(self._mode15_l2_to_l1(sequences, labels, return_details))


        raise ValueError(f"Unsupported mode: {self.current_mode_id}")

    # =========================================================================
    # MODE 1/2/3: Individual
    # =========================================================================
    def _mode_individual(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        layer_name = self.current_mode["layers"][0]

        t0 = time.time()
        # LLM-based layers run sample-by-sample; BERT layers use larger batches
        if self._is_llm_layer(layer_name):
            bs = 1
        elif layer_name == "layer2":
            bs = 4
        else:
            bs = 128
        out = self._run_layer(layer_name, sequences, batch_size=bs)
        elapsed = time.time() - t0

        preds = out["predictions"]
        probs = out["probabilities"]

        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, preds, probs)
            if probs is not None:
                metrics["roc_curve"] = compute_roc_curve_data(labels, probs)
                metrics["pr_curve"] = compute_precision_recall_curve_data(labels, probs)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "individual",
            "predictions": preds,
            "num_samples": n,
            "num_detected": int(preds.sum()),
            "time": elapsed,
            "layer_stats": {layer_name: {"processed": n, "detected": int(preds.sum()), "time": elapsed}},
            "probabilities": probs
        }
        self._attach_layer_confusion(res["layer_stats"], layer_name, labels, preds)
        if layer_name == "layer1":
            res["layer1_embedding_cache_loaded"] = bool(out["raw"].get("used_cached_embeddings", False))
            res["layer1_embedding_batch_size"] = int(out["raw"].get("embedding_batch_size", bs))

        if metrics is not None:
            res["metrics"] = metrics

        # if return_details:
        #     res["probabilities"] = probs

        return res

    # =========================================================================
    # MODE 10: L1 -> L2
    # L1: pred=0 forward (done), pred=1 suspicious -> L2
    # L2: pred=1 block, pred=0 forward
    # =========================================================================
    def _mode10_l1_to_l2(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        scores = np.zeros(n, dtype=float)
        final = np.zeros(n, dtype=int)
        layer_stats: Dict[str, Dict[str, Any]] = {}

        t0 = time.time()

        # L1 on all
        out1 = self._run_layer("layer1", sequences, batch_size=128)
        p1 = out1["predictions"]
        idx_susp = np.where(p1 == 1)[0]
        s1 = out1["probabilities"]
        scores[:] = s1

        layer_stats["layer1"] = {
            "processed": n,
            "forwarded": int((p1 == 0).sum()),
            "sent_to_next": int((p1 == 1).sum()),
        }
        self._attach_layer_confusion(layer_stats, "layer1", labels, p1)

        # L2 on suspicious only
        if len(idx_susp) > 0:
            sub2 = [sequences[i] for i in idx_susp]
            out2 = self._run_layer("layer2", sub2, batch_size=4)
            p2 = out2["predictions"]
            s2 = out2["probabilities"]
            scores[idx_susp] = s2

            idx_block = idx_susp[p2 == 1]
            final[idx_block] = 1

            layer_stats["layer2"] = {
                "processed": len(idx_susp),
                "blocked": int((p2 == 1).sum()),
                "forwarded": int((p2 == 0).sum()),
            }
            labels2 = labels[idx_susp] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer2", labels2, p2)
        else:
            layer_stats["layer2"] = {"processed": 0, "blocked": 0, "forwarded": 0}

        elapsed = time.time() - t0

        # metrics = compute_complete_metrics(labels, final) if labels is not None else None
        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)
        flow_data = analyze_flow(layer_stats, n)
        flow_diagram = generate_flow_diagram_data(layer_stats, n)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "chained",
            "predictions": final,
            "num_samples": n,
            "num_detected": int(final.sum()),
            "time": elapsed,
            "layer_stats": layer_stats,
            "flow_analysis": flow_data,
            "flow_diagram": flow_diagram,
            "probabilities": scores
        }
        res["layer1_embedding_cache_loaded"] = bool(out1["raw"].get("used_cached_embeddings", False))
        res["layer1_embedding_batch_size"] = int(out1["raw"].get("embedding_batch_size", 128))
        if metrics is not None:
            res["metrics"] = metrics
        if return_details:
            res["debug"] = {"idx_suspicious_from_l1": idx_susp.tolist()}
        return res

    # =========================================================================
    # MODE 11: L1 -> L2 -> L3
    # L1: pred=0 forward (done), pred=1 suspicious -> L2
    # L2: pred=1 block (done), pred=0 -> L3
    # L3: final decision for remaining
    # =========================================================================
    def _mode11_l1_to_l2_to_l3(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        scores = np.zeros(n, dtype=float)
        final = np.zeros(n, dtype=int)
        layer_stats: Dict[str, Dict[str, Any]] = {}

        t0 = time.time()

        # L1 on all
        out1 = self._run_layer("layer1", sequences, batch_size=128)
        p1 = out1["predictions"]
        s1 = out1["probabilities"]
        scores[:] = s1
        idx_susp = np.where(p1 == 1)[0] # Only get the suspicious part of the data

        layer_stats["layer1"] = {
            "processed": n,
            "forwarded": int((p1 == 0).sum()),
            "sent_to_next": int((p1 == 1).sum()),
        }
        self._attach_layer_confusion(layer_stats, "layer1", labels, p1)

        # L2 on suspicious
        idx_to_l3 = np.array([], dtype=int)
        if len(idx_susp) > 0:
            sub2 = [sequences[i] for i in idx_susp]
            out2 = self._run_layer("layer2", sub2, batch_size=4)
            p2 = out2["predictions"]
            s2 = out2["probabilities"]
            scores[idx_susp] = s2

            # block malicious
            idx_block = idx_susp[p2 == 1]
            final[idx_block] = 1

            # send non-malicious to L3
            idx_to_l3 = idx_susp[p2 == 0]

            # tau_low = self.L2_TAU_LOW
            # tau_high = self.L2_TAU_HIGH

            # # confident malicious
            # idx_block = idx_susp[s2 >= tau_high]

            # # confident benign
            # idx_forward = idx_susp[s2 <= tau_low]

            # # uncertain
            # idx_to_l3 = idx_susp[(s2 > tau_low) & (s2 < tau_high)]

            # final[idx_block] = 1

            layer_stats["layer2"] = {
                "processed": len(idx_susp),
                "blocked": int((p2 == 1).sum()),
                "sent_to_next": int((p2 == 0).sum()),
            }
            labels2 = labels[idx_susp] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer2", labels2, p2)
            # layer_stats["layer2"] = {
            #     "processed": len(idx_susp),
            #     "blocked": len(idx_block),
            #     "forwarded": len(idx_forward),
            #     "sent_to_next": len(idx_to_l3),
            # }
        else:
            layer_stats["layer2"] = {"processed": 0, "blocked": 0, "sent_to_next": 0}

        # L3 on remaining from L2
        if len(idx_to_l3) > 0:
            sub3 = [sequences[i] for i in idx_to_l3]
            out3 = self._run_layer("layer3", sub3, batch_size=1)  # LLM often 1-by-1
            p3 = out3["predictions"]
            s3 = out3["probabilities"]
            scores[idx_to_l3] = s3

            final[idx_to_l3[p3 == 1]] = 1

            layer_stats["layer3"] = {
                "processed": len(idx_to_l3),
                "blocked": int((p3 == 1).sum()),
                "forwarded": int((p3 == 0).sum()),
            }
            labels3 = labels[idx_to_l3] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer3", labels3, p3)
        else:
            layer_stats["layer3"] = {"processed": 0, "blocked": 0, "forwarded": 0}

        elapsed = time.time() - t0

        # metrics = compute_complete_metrics(labels, final) if labels is not None else None
        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"]  = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

        flow_data = analyze_flow(layer_stats, n)
        flow_diagram = generate_flow_diagram_data(layer_stats, n)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "chained",
            "predictions": final,
            "num_samples": n,
            "num_detected": int(final.sum()),
            "time": elapsed,
            "layer_stats": layer_stats,
            "flow_analysis": flow_data,
            "flow_diagram": flow_diagram,
            "probabilities": scores
        }
        res["layer1_embedding_cache_loaded"] = bool(out1["raw"].get("used_cached_embeddings", False))
        res["layer1_embedding_batch_size"] = int(out1["raw"].get("embedding_batch_size", 128))
        if metrics is not None:
            res["metrics"] = metrics
        if return_details:
            res["debug"] = {
                "idx_suspicious_from_l1": idx_susp.tolist(),
                "idx_sent_to_l3_from_l2": idx_to_l3.tolist(),
            }
        return res

    # =========================================================================
    # MODE 12: L2 -> L1 -> L3
    # L2: pred=1 block (done), pred=0 -> L1
    # L1: pred=0 forward (done), pred=1 suspicious -> L3
    # L3: final decision for remaining
    # =========================================================================
    def _mode12_l2_to_l1_to_l3(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        scores = np.zeros(n, dtype=float)
        final = np.zeros(n, dtype=int)
        layer_stats: Dict[str, Dict[str, Any]] = {}

        t0 = time.time()

        # L2 on all
        out2 = self._run_layer("layer2", sequences, batch_size=4)
        p2 = out2["predictions"]
        s2 = out2["probabilities"]
        scores[:] = s2   

        idx_block = np.where(p2 == 1)[0]
        final[idx_block] = 1  # block immediately

        idx_to_l1 = np.where(p2 == 0)[0]

        # tau_low = self.L2_TAU_LOW
        # tau_high = self.L2_TAU_HIGH

        # idx_block = np.where(s2 >= tau_high)[0]
        # idx_forward = np.where(s2 <= tau_low)[0]
        # idx_to_l1 = np.where((s2 > tau_low) & (s2 < tau_high))[0]

        # final[idx_block] = 1

        layer_stats["layer2"] = {
            "processed": n,
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
        }
        self._attach_layer_confusion(layer_stats, "layer2", labels, p2)
        
        # layer_stats["layer2"] = {
        #     "processed": n,
        #     "blocked": len(idx_block),
        #     "forwarded": len(idx_forward),
        #     "sent_to_next": len(idx_to_l1),
        # }

        # L1 on not-malicious
        idx_to_l3 = np.array([], dtype=int)
        if len(idx_to_l1) > 0:
            sub1 = [sequences[i] for i in idx_to_l1]
            out1 = self._run_layer("layer1", sub1, batch_size=128)
            p1 = out1["predictions"]
            s1 = out1["probabilities"]
            scores[idx_to_l1] = s1

            # pred=1 suspicious -> L3
            idx_to_l3 = idx_to_l1[p1 == 1]

            layer_stats["layer1"] = {
                "processed": len(idx_to_l1),
                "forwarded": int((p1 == 0).sum()),
                "sent_to_next": int((p1 == 1).sum()),
            }
            labels1 = labels[idx_to_l1] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer1", labels1, p1)
        else:
            layer_stats["layer1"] = {"processed": 0, "forwarded": 0, "sent_to_next": 0}

        # L3 on suspicious-from-L1
        if len(idx_to_l3) > 0:
            sub3 = [sequences[i] for i in idx_to_l3]
            out3 = self._run_layer("layer3", sub3, batch_size=1)
            p3 = out3["predictions"]
            s3 = out3["probabilities"]
            scores[idx_to_l3] = s3
            final[idx_to_l3[p3 == 1]] = 1

            layer_stats["layer3"] = {
                "processed": len(idx_to_l3),
                "blocked": int((p3 == 1).sum()),
                "forwarded": int((p3 == 0).sum()),
            }
            labels3 = labels[idx_to_l3] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer3", labels3, p3)
        else:
            layer_stats["layer3"] = {"processed": 0, "blocked": 0, "forwarded": 0}

        elapsed = time.time() - t0

        # metrics = compute_complete_metrics(labels, final) if labels is not None else None
        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

        flow_data = analyze_flow(layer_stats, n)
        flow_diagram = generate_flow_diagram_data(layer_stats, n)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "chained",
            "predictions": final,
            "num_samples": n,
            "num_detected": int(final.sum()),
            "time": elapsed,
            "layer_stats": layer_stats,
            "flow_analysis": flow_data,
            "flow_diagram": flow_diagram,
            "probabilities": scores
        }
        if len(idx_to_l1) > 0:
            res["layer1_embedding_cache_loaded"] = bool(out1["raw"].get("used_cached_embeddings", False))
            res["layer1_embedding_batch_size"] = int(out1["raw"].get("embedding_batch_size", 128))
        if metrics is not None:
            res["metrics"] = metrics
        if return_details:
            res["debug"] = {
                "idx_blocked_at_l2": idx_block.tolist(),
                "idx_sent_to_l1_from_l2": idx_to_l1.tolist(),
                "idx_sent_to_l3_from_l1": idx_to_l3.tolist(),
            }
        return res

    # =========================================================================
    # MODE 13: L1 -> L3
    # L1: pred=0 forward (done), pred=1 suspicious -> L3
    # L3: final decision for suspicious samples
    # =========================================================================
    def _mode13_l1_to_l3(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        scores = np.zeros(n, dtype=float)
        final = np.zeros(n, dtype=int)
        layer_stats: Dict[str, Dict[str, Any]] = {}

        t0 = time.time()

        out1 = self._run_layer("layer1", sequences, batch_size=128)
        p1 = out1["predictions"]
        s1 = out1["probabilities"]
        scores[:] = s1
        idx_to_l3 = np.where(p1 == 1)[0]

        layer_stats["layer1"] = {
            "processed": n,
            "forwarded": int((p1 == 0).sum()),
            "sent_to_next": int((p1 == 1).sum()),
        }
        self._attach_layer_confusion(layer_stats, "layer1", labels, p1)

        if len(idx_to_l3) > 0:
            sub3 = [sequences[i] for i in idx_to_l3]
            out3 = self._run_layer("layer3", sub3, batch_size=1)
            p3 = out3["predictions"]
            s3 = out3["probabilities"]
            scores[idx_to_l3] = s3
            final[idx_to_l3[p3 == 1]] = 1

            layer_stats["layer3"] = {
                "processed": len(idx_to_l3),
                "blocked": int((p3 == 1).sum()),
                "forwarded": int((p3 == 0).sum()),
            }
            labels3 = labels[idx_to_l3] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer3", labels3, p3)
        else:
            layer_stats["layer3"] = {"processed": 0, "blocked": 0, "forwarded": 0}

        elapsed = time.time() - t0

        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

        flow_data = analyze_flow(layer_stats, n)
        flow_diagram = generate_flow_diagram_data(layer_stats, n)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "chained",
            "predictions": final,
            "num_samples": n,
            "num_detected": int(final.sum()),
            "time": elapsed,
            "layer_stats": layer_stats,
            "flow_analysis": flow_data,
            "flow_diagram": flow_diagram,
            "probabilities": scores,
        }
        res["layer1_embedding_cache_loaded"] = bool(out1["raw"].get("used_cached_embeddings", False))
        res["layer1_embedding_batch_size"] = int(out1["raw"].get("embedding_batch_size", 128))
        if metrics is not None:
            res["metrics"] = metrics
        if return_details:
            res["debug"] = {"idx_sent_to_l3_from_l1": idx_to_l3.tolist()}
        return res

    # =========================================================================
    # MODE 14: L2 -> L3
    # L2: pred=1 block (done), pred=0 -> L3
    # L3: final decision for remaining samples
    # =========================================================================
    def _mode14_l2_to_l3(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        scores = np.zeros(n, dtype=float)
        final = np.zeros(n, dtype=int)
        layer_stats: Dict[str, Dict[str, Any]] = {}

        t0 = time.time()

        out2 = self._run_layer("layer2", sequences, batch_size=4)
        p2 = out2["predictions"]
        s2 = out2["probabilities"]
        scores[:] = s2

        idx_block = np.where(p2 == 1)[0]
        idx_to_l3 = np.where(p2 == 0)[0]
        final[idx_block] = 1

        layer_stats["layer2"] = {
            "processed": n,
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
        }
        self._attach_layer_confusion(layer_stats, "layer2", labels, p2)

        if len(idx_to_l3) > 0:
            sub3 = [sequences[i] for i in idx_to_l3]
            out3 = self._run_layer("layer3", sub3, batch_size=1)
            p3 = out3["predictions"]
            s3 = out3["probabilities"]
            scores[idx_to_l3] = s3
            final[idx_to_l3[p3 == 1]] = 1

            layer_stats["layer3"] = {
                "processed": len(idx_to_l3),
                "blocked": int((p3 == 1).sum()),
                "forwarded": int((p3 == 0).sum()),
            }
            labels3 = labels[idx_to_l3] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer3", labels3, p3)
        else:
            layer_stats["layer3"] = {"processed": 0, "blocked": 0, "forwarded": 0}

        elapsed = time.time() - t0

        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

        flow_data = analyze_flow(layer_stats, n)
        flow_diagram = generate_flow_diagram_data(layer_stats, n)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "chained",
            "predictions": final,
            "num_samples": n,
            "num_detected": int(final.sum()),
            "time": elapsed,
            "layer_stats": layer_stats,
            "flow_analysis": flow_data,
            "flow_diagram": flow_diagram,
            "probabilities": scores,
        }
        if metrics is not None:
            res["metrics"] = metrics
        if return_details:
            res["debug"] = {
                "idx_blocked_at_l2": idx_block.tolist(),
                "idx_sent_to_l3_from_l2": idx_to_l3.tolist(),
            }
        return res

    # =========================================================================
    # MODE 15: L2 -> L1
    # L2: pred=1 block (done), pred=0 -> L1
    # L1: final decision for remaining samples
    # =========================================================================
    def _mode15_l2_to_l1(
        self,
        sequences: List[str],
        labels: Optional[np.ndarray],
        return_details: bool,
    ) -> Dict[str, Any]:
        n = len(sequences)
        scores = np.zeros(n, dtype=float)
        final = np.zeros(n, dtype=int)
        layer_stats: Dict[str, Dict[str, Any]] = {}

        t0 = time.time()

        out2 = self._run_layer("layer2", sequences, batch_size=4)
        p2 = out2["predictions"]
        s2 = out2["probabilities"]
        scores[:] = s2

        idx_block = np.where(p2 == 1)[0]
        idx_to_l1 = np.where(p2 == 0)[0]
        final[idx_block] = 1

        layer_stats["layer2"] = {
            "processed": n,
            "blocked": int((p2 == 1).sum()),
            "sent_to_next": int((p2 == 0).sum()),
        }
        self._attach_layer_confusion(layer_stats, "layer2", labels, p2)

        if len(idx_to_l1) > 0:
            sub1 = [sequences[i] for i in idx_to_l1]
            out1 = self._run_layer("layer1", sub1, batch_size=128)
            p1 = out1["predictions"]
            s1 = out1["probabilities"]
            scores[idx_to_l1] = s1
            final[idx_to_l1[p1 == 1]] = 1

            layer_stats["layer1"] = {
                "processed": len(idx_to_l1),
                "blocked": int((p1 == 1).sum()),
                "forwarded": int((p1 == 0).sum()),
            }
            labels1 = labels[idx_to_l1] if labels is not None else None
            self._attach_layer_confusion(layer_stats, "layer1", labels1, p1)
        else:
            layer_stats["layer1"] = {"processed": 0, "blocked": 0, "forwarded": 0}

        elapsed = time.time() - t0

        metrics = None
        if labels is not None:
            metrics = compute_complete_metrics(labels, final, scores)
            metrics["roc_curve"] = compute_roc_curve_data(labels, scores)
            metrics["pr_curve"] = compute_precision_recall_curve_data(labels, scores)

        flow_data = analyze_flow(layer_stats, n)
        flow_diagram = generate_flow_diagram_data(layer_stats, n)

        res: Dict[str, Any] = {
            "test_mode": self.current_mode.get("name", ""),
            "test_mode_id": self.current_mode_id,
            "strategy": "chained",
            "predictions": final,
            "num_samples": n,
            "num_detected": int(final.sum()),
            "time": elapsed,
            "layer_stats": layer_stats,
            "flow_analysis": flow_data,
            "flow_diagram": flow_diagram,
            "probabilities": scores,
        }
        if len(idx_to_l1) > 0:
            res["layer1_embedding_cache_loaded"] = bool(out1["raw"].get("used_cached_embeddings", False))
            res["layer1_embedding_batch_size"] = int(out1["raw"].get("embedding_batch_size", 128))
        if metrics is not None:
            res["metrics"] = metrics
        if return_details:
            res["debug"] = {
                "idx_blocked_at_l2": idx_block.tolist(),
                "idx_sent_to_l1_from_l2": idx_to_l1.tolist(),
            }
        return res

    # -------------------------
    # Convenience
    # -------------------------
    def detect_single(self, sequence: str, label: Optional[int] = None) -> Tuple[bool, Dict[str, Any]]:
        labels = np.array([label], dtype=int) if label is not None else None
        r = self.predict([sequence], labels=labels, return_details=True)
        return bool(int(r["predictions"][0])), r

    def list_test_modes(self) -> str:
        lines = ["\n" + "=" * 80, "AVAILABLE TEST MODES", "=" * 80]
        for mode_id, mode in sorted(self.all_test_modes.items()):
            status = "✅" if mode.get("enabled", True) else "❌"
            lines.append(f"\n{status} Mode {mode_id}: {mode.get('name', '')}")
            lines.append(f"   Layers: {' → '.join(mode.get('layers', []))}")
        lines.append("\n" + "=" * 80)
        return "\n".join(lines)

    def __call__(self, sequences: List[str], labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        return self.predict(sequences, labels=labels)
    
    # Add more function 
    def find_threshold_no_fn(
        self,
        sequences: List[str],
        labels: np.ndarray,
    ):
        """
        Find the smallest Layer1 threshold that achieves FN = 0
        while minimizing FP.
        """

        if "layer1" not in self.layers:
            raise ValueError("Layer1 not loaded in this test mode.")

        layer1 = self.layers["layer1"]

        # Save original threshold
        original_threshold = layer1.threshold

        # Get AE errors
        out = layer1.predict(sequences, batch_size=128)
        errors = np.asarray(out["errors"])

        # Candidate thresholds from real distribution
        candidate_thresholds = np.sort(errors)

        best_threshold = None
        best_fp = None

        for t in candidate_thresholds:

            preds = (errors > t).astype(int)

            FN = np.sum((labels == 1) & (preds == 0))
            FP = np.sum((labels == 0) & (preds == 1))

            if FN == 0:

                if best_fp is None or FP < best_fp:
                    best_threshold = t
                    best_fp = FP

        # Restore original threshold
        layer1.threshold = original_threshold

        return {
            "threshold": best_threshold,
            "fp": int(best_fp) if best_fp is not None else None
        }
