# ============================================================================
# FILE: models/detectors/layer2_detector.py
# Layer 2 supervised classifier for known malicious trace patterns.
#
# Purpose:
# - Applies a fine-tuned sequence classifier to determine whether suspicious
#   traces should be blocked as malicious.
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


class Layer2Detector:
    """
    Layer 2: Fine-tuned BERT classifier
    Receives samples that passed Layer 1 (low reconstruction error)
    """
    
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        threshold: float = 0.5
    ):
        """
        Initialize Layer 2 detector
        
        Args:
            model_path: Path to fine-tuned BERT classifier (same as Layer 1 encoder)
            device: 'cuda' or 'cpu'
            threshold: Classification threshold
        """
        self.model_path = Path(model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        
        # Load config
        with open(self.model_path / "training_config.json", 'r') as f:
            self.config = json.load(f)
        
        self.max_length = self.config["max_length"]
        self.trust_remote_code = self.config.get("trust_remote_code", False)
        
        # Load model
        self._load_model()
        
        print(f"✅ Layer 2 loaded (Classifier)")
        print(f"   Model: {self.config['model_name']}")
        print(f"   Device: {self.device}")
        print(f"   Threshold: {self.threshold}")
    
    def _load_model(self):
        """Load classifier model"""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            trust_remote_code=self.trust_remote_code
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
                
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = (probs > self.threshold).astype(int)
                
                all_predictions.extend(preds)
                all_probs.extend(probs)
        
        predictions = np.array(all_predictions)
        probabilities = np.array(all_probs)
        
        result = {
            "predictions": predictions,
            "num_detected": int(predictions.sum()),
            "detection_rate": float(predictions.mean()),
            "layer": "Layer2"
        }
        
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
