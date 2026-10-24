# ============================================================================
# FILE: models/detectors/gate.py
# Gate anomaly detector built from encoder embeddings and an autoencoder.
#
# Purpose:
# - Screens traces for unfamiliar behavior using reconstruction error over
#   embedding vectors from the fine-tuned encoder.
#
# Workflow:
# - Loads tokenizer and encoder artifacts.
# - Extracts CLS embeddings for each sequence.
# - Reconstructs embeddings with the trained autoencoder.
# - Flags high-error samples as suspicious for escalation.
#
# Use this file when:
# - You want to inspect anomaly scoring or embedding extraction behavior.
# - You need to adjust thresholding or embedding-cache handling.
# ============================================================================
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModel
import json
from contextlib import nullcontext
from tqdm.auto import tqdm

class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim, dropout):
        super().__init__()

        # Encoder
        encoder_layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        decoder_layers.append(nn.Linear(in_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self,x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent
    
class GateDetector:
    def __init__(
        self,
        bert_model_path: str,
        ae_model_path: str,
        device: str = "cuda",
        threshold: float = 0.1
    ):
        self.bert_model_path = Path(bert_model_path)
        self.ae_model_path = Path(ae_model_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        with open(self.bert_model_path / "training_config.json", "r") as f:
            self.bert_config = json.load(f)

        self.max_length = self.bert_config["max_length"]
        self.trust_remote_code = self.bert_config.get("trust_remote_code", False)
        self.embedding_precision = self.bert_config.get("embedding_precision", "float32")

        self._load_models()

        print(f"✅ Gate loaded (Autoencoder)")
        print(f"   BERT encoder: {self.bert_config['model_name']}")
        print(f"   Device: {self.device}")
        print(f"   Threshold: {self.threshold:.6f}")

    def _load_models(self):
        # Load BERT and AE
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.bert_model_path,
            trust_remote_code = self.trust_remote_code
        )
        model_options = ({"attn_implementation": self.bert_config["attention_implementation"],
                          "reference_compile": False} if "attention_implementation" in self.bert_config else {})
        self.embedding_model = AutoModel.from_pretrained(
            self.bert_model_path,
            trust_remote_code = self.trust_remote_code,
            **model_options,
        )
        self.embedding_model = self.embedding_model.to(self.device)
        # frozen the weights to evaluate
        self.embedding_model.eval()

        for param in self.embedding_model.parameters():
            param.requires_grad = False

        # Load AE
        ae_checkpoint = torch.load(self.ae_model_path, map_location = self.device)
        from research_runner.implementation.training.gate_common import directory_hash
        self.encoder_fingerprint = directory_hash(self.bert_model_path)
        if ae_checkpoint.get("encoder_sha256") and ae_checkpoint["encoder_sha256"] != self.encoder_fingerprint:
            raise ValueError("AE checkpoint and encoder fingerprint do not match")
        ae_config = ae_checkpoint["config"]

        self.autoencoder = AutoEncoder(
            input_dim = ae_config["input_dim"],
            hidden_dims=ae_config["hidden_dims"],
            latent_dim=ae_config["latent_dim"],
            dropout=ae_config["dropout"]
        )

        self.autoencoder.load_state_dict(ae_checkpoint["model_state_dict"])
        self.autoencoder = self.autoencoder.to(self.device)
        self.autoencoder.eval()

        for param in self.autoencoder.parameters():
            param.requires_grad = False
        
    # def extract_embedding(
    #     self,
    #     sequences: List[str],
    #     batch_size: int = 32
    # ) -> np.ndarray:
    #     # Extract embedding using fine-tuned BERT
    #     embeddings = []

    #     with torch.no_grad():
    #         for i in tqdm(range(0, len(sequences), batch_size), desc="Gate: Embedding", unit="batch"):
    #             batch = sequences[i: i+batch_size]
    #             inputs = self.tokenizer(
    #                 batch, 
    #                 max_length = self.max_length,
    #                 truncation = True,
    #                 padding = "max_length",
    #                 return_tensors = "pt"
    #             )
    #             inputs = {k: v.to(self.device) for k,v in inputs.items()}
    #             outputs = self.embedding_model(**inputs)
    #             cls_embeddings = outputs.last_hidden_state[:,0,:]
    #             embeddings.append(cls_embeddings.cpu().numpy())
    #     return np.vstack(embeddings)
    
    def extract_embedding(
        self,
        sequences: List[str],
        batch_size: int = 32,
        cache_dir: str = "results/research_runner/gate_embedding_cache"
    ) -> np.ndarray:

        import hashlib
        from pathlib import Path

        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        # 🔑 tạo fingerprint từ dữ liệu + model đang encode
        # (bert_model_path bắt buộc phải nằm trong hash: nếu không, hai
        # checkpoint ModernBERT khác nhau nhưng xử lý cùng một tập input sẽ
        # đụng chung 1 file cache, khiến embedding của model A bị dùng nhầm
        # cho model B -- đây là nguyên nhân gây sai lệch âm thầm và crash
        # shape-mismatch khi chạy Std Testing với nhiều checkpoint song song.)
        hash_input = json.dumps({"encoder": self.encoder_fingerprint, "max_length": self.max_length,
                                 "precision": self.embedding_precision, "sequences": sequences}, ensure_ascii=False)
        dataset_hash = hashlib.sha256(hash_input.encode()).hexdigest()

        cache_path = cache_dir / f"emb_{dataset_hash}.npy"
        self.last_embedding_cache_loaded = False
        batch_size = min(batch_size, int(self.bert_config.get("embedding_batch_size", batch_size)))
        self.last_embedding_batch_size = int(batch_size)

        # -------------------------
        # LOAD nếu có
        # -------------------------
        if cache_path.exists():
            cached = np.load(cache_path)
            if cached.shape == (len(sequences), self.autoencoder.encoder[0].in_features):
                self.last_embedding_cache_loaded = True
                print(f"⚡ Loading cached embeddings: {cache_path}")
                return cached

        # -------------------------
        # COMPUTE nếu chưa có
        # -------------------------
        print("🔄 Computing embeddings...")

        embeddings = []

        with torch.no_grad():
            for i in tqdm(range(0, len(sequences), batch_size), desc="Gate: Embedding", unit="batch"):
                batch = sequences[i: i+batch_size]
                inputs = self.tokenizer(
                    batch,
                    max_length=self.max_length,
                    truncation=True,
                    padding=True if self.bert_config.get("padding") == "dynamic" else "max_length",
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                context = (torch.autocast("cuda", dtype=torch.bfloat16)
                           if self.device.type == "cuda" and self.embedding_precision == "bf16" else nullcontext())
                with context:
                    outputs = self.embedding_model(**inputs)

                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings.float().cpu().numpy())

        embeddings = np.vstack(embeddings)

        # -------------------------
        # SAVE
        # -------------------------
        np.save(cache_path, embeddings)
        print(f"💾 Saved embeddings to {cache_path}")

        return embeddings
    
    def compute_reconstruction_errors(self, embeddings: np.ndarray) -> np.ndarray:
        errors = []
        batch_size = 128
        with torch.no_grad():
            for i in tqdm(range(0, len(embeddings), batch_size), desc="Gate: AE Reconstructing", unit="batch"):
                batch = torch.FloatTensor(embeddings[i:i+batch_size]).to(self.device)
                reconstructed, _ = self.autoencoder(batch)
                mse = torch.mean((batch-reconstructed)**2, dim=1)
                errors.append(mse.cpu().numpy())
        return np.concatenate(errors)

    def predict(
        self,
        sequences: List[str],
        batch_size: int = 32,
        return_errors: bool = True
    ) -> Dict:
        embeddings = self.extract_embedding(sequences, batch_size)
        errors = self.compute_reconstruction_errors(embeddings=embeddings)
        predictions = (errors > self.threshold).astype(int)
        result = {
            "predictions": predictions,
            "num_detected": int(predictions.sum()),
            "detection_rate": float(predictions.mean()),
            "layer": "Gate",
            "used_cached_embeddings": bool(getattr(self, "last_embedding_cache_loaded", False)),
            "embedding_batch_size": int(getattr(self, "last_embedding_batch_size", batch_size)),
        }
        if return_errors:
            result["errors"] = errors
        return result
    
    def detect_single(self, sequence: str) -> Tuple[bool, float]:
        result = self.predict([sequence], return_errors=True)
        return bool(result["predictions"][0]), float(result["errors"][0])
    
    def __call__(self, sequences: List[str]) -> Dict:
        return self.predict(sequences=sequences)
