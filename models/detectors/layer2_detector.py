# ============================================================================
# FILE: models/detectors/layer2_detector.py
# Layer 2 supervised classifier for known malicious trace patterns.
#
# Purpose:
# - Supports frozen Hugging Face source-channel inference and legacy checkpoints.
#
# Workflow:
# - Loads the tokenizer and classification model from a saved checkpoint.
# - Tokenizes sequences in batches.
# - Produces malicious probabilities and binary predictions.
#
# Use this file when:
# - You want to inspect Layer 2 threshold behavior or classification outputs.
# - You need to switch or debug the fine-tuned classifier checkpoint.
# ============================================================================
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
from tqdm.auto import tqdm
from .channel_payloads import channel_payloads


class Layer2Detector:
    """
    Frozen classifier; optionally inspect source channels independently.
    In the full cascade, receives suspicious (high-error) Layer 1 samples.
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        threshold: float = 0.5,
        channel_wise: bool = False,
        revision: str = None,
        window_overlap: int = 64,
        local_files_only: bool = False,
    ):
        """
        Initialize Layer 2 detector
        
        Args:
            model_path: Hugging Face model ID or local classifier checkpoint
            device: 'cuda' or 'cpu'
            threshold: Classification threshold
        """
        self.model_path = str(model_path)
        self.channel_wise = channel_wise
        self.revision = revision
        self.local_files_only = local_files_only
        self.window_overlap = window_overlap
        if not 0 <= threshold <= 1:
            raise ValueError("Layer 2 threshold must be in [0, 1]")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # Load config
        training_config = Path(self.model_path) / "training_config.json"
        self.config = json.loads(training_config.read_text()) if training_config.exists() else {
            "model_name": self.model_path, "max_length": 512}
        
        self.max_length = self.config["max_length"]
        self.trust_remote_code = self.config.get("trust_remote_code", False)
        
        # Load model
        self._load_model()
        labels = self.model.config.id2label
        injection_ids = [int(i) for i, name in labels.items() if str(name).upper() == "INJECTION"]
        if self.channel_wise and len(injection_ids) != 1:
            raise ValueError("Channel-wise L2 requires an explicit INJECTION class mapping")
        self.malicious_index = injection_ids[0] if injection_ids else 1
        self.max_length = min(self.max_length, self.model.config.max_position_embeddings)
        if not 0 <= window_overlap < self.max_length - self.tokenizer.num_special_tokens_to_add(False):
            raise ValueError("Invalid token-window overlap")
        
        print(f"✅ Layer 2 loaded (Classifier)")
        print(f"   Model: {self.config['model_name']}")
        print(f"   Device: {self.device}")
        print(f"   Threshold: {self.threshold}")
    
    def _load_model(self):
        """Load classifier model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision, local_files_only=self.local_files_only,
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code,
            revision=self.revision, local_files_only=self.local_files_only,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        for param in self.model.parameters():
            param.requires_grad = False
    
    def predict(
        self,
        sequences: List[str],
        batch_size: int = 32,
        return_probs: bool = True
    ) -> Dict:
        """
        Predict using BERT classifier
        
        Returns:
            Dict with predictions and statistics
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.channel_wise:
            return self._predict_channels(sequences, batch_size, return_probs)
        all_predictions = []
        all_probs = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Layer 2: Classify", unit="batch"):
                batch = sequences[i:i+batch_size]
                
                inputs = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                probs = torch.softmax(logits, dim=1)[:, self.malicious_index].cpu().numpy()
                preds = (probs > self.threshold).astype(int)
                
                all_predictions.extend(preds)
                all_probs.extend(probs)
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probs)
        
        result = {
            "predictions": predictions,
            "num_detected": int(predictions.sum()),
            "detection_rate": float(predictions.mean()) if len(predictions) else 0.0,
            "layer": "Layer2"
        }
        
        if return_probs:
            result["probabilities"] = probabilities
        
        return result

    def _predict_channels(self, sequences, batch_size, return_probs):
        payloads = [channel_payloads(sequence) for sequence in sequences]
        unique = list(dict.fromkeys(text for parts in payloads for _, text in parts))
        windows, owners = [], []
        for owner, text in enumerate(unique):
            encoded = self.tokenizer(text, max_length=self.max_length, truncation=True,
                                     stride=self.window_overlap, return_overflowing_tokens=True,
                                     padding=False)
            for index in range(len(encoded["input_ids"])):
                windows.append({key: encoded[key][index] for key in self.tokenizer.model_input_names if key in encoded})
                owners.append(owner)
        scores = np.zeros(len(unique), dtype=float)
        with torch.inference_mode():
            for start in tqdm(range(0, len(windows), batch_size), desc="Layer 2: source windows", unit="batch"):
                inputs = self.tokenizer.pad(windows[start:start + batch_size], padding=True, return_tensors="pt")
                logits = self.model(**{key: value.to(self.device) for key, value in inputs.items()}).logits
                probabilities = torch.softmax(logits.float(), dim=-1)[:, self.malicious_index].cpu().numpy()
                np.maximum.at(scores, owners[start:start + batch_size], probabilities)
        lookup = dict(zip(unique, scores))
        channel_scores = []
        for parts in payloads:
            channels = {name: 0.0 for name in ("prompt", "retrieval", "tool")}
            for name, text in parts:
                channels[name] = max(channels[name], float(lookup[text]))
            channel_scores.append(channels)
        probabilities = np.array([max(channels.values()) for channels in channel_scores])
        predictions = (probabilities > self.threshold).astype(int)
        result = {"predictions": predictions, "num_detected": int(predictions.sum()),
                  "detection_rate": float(predictions.mean()) if len(predictions) else 0.0,
                  "layer": "Layer2", "channel_probabilities": channel_scores,
                  "unique_payloads": len(unique), "windows": len(windows)}
        if return_probs:
            result["probabilities"] = probabilities
        return result
    
    def detect_single(self, sequence: str) -> Tuple[bool, float]:
        """Detect single sequence"""
        result = self.predict([sequence], return_probs=True)
        return bool(result["predictions"][0]), float(result["probabilities"][0])
    
    def __call__(self, sequences: List[str]) -> Dict:
        """Shortcut for predict"""
        return self.predict(sequences)
