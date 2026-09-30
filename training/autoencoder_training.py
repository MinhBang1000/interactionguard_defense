# ============================================================================
# FILE: training/autoencoder_training.py
# Train the Layer 1 autoencoder on embeddings from the fine-tuned encoder.
#
# Purpose:
# - Learns the benign embedding distribution used by the anomaly detector in
#   Layer 1.
#
# Workflow:
# - Loads benign processed data.
# - Extracts encoder embeddings.
# - Trains an autoencoder to reconstruct benign embeddings well.
# - Saves the checkpoint used by runtime Layer 1 detection.
#
# Use this file when:
# - You want to rebuild the anomaly detector after updating the encoder.
# ============================================================================
# ============================================================================
# Minimal Autoencoder Training
# Train AE on embeddings from fine-tuned encoder
# Keeps ALL original configuration and behavior
# ============================================================================

import os

import torch
import copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
from utils.utils import load_jsonl
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
from config.config import AutoEncoderConfig

config = AutoEncoderConfig()
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.CUDA_VISIBLE_DEVICES)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(config.SEED)
np.random.seed(config.SEED)


# ============================================================================
# DATA
# ============================================================================

train_data = load_jsonl(config.PROCESSED_DIR / config.TRAIN_FILE)
val_data   = load_jsonl(config.PROCESSED_DIR / config.VAL_FILE)
test_data  = load_jsonl(config.PROCESSED_DIR / config.TEST_FILE)

train_benign = [d for d in train_data if d["label"] == 0]
val_benign   = [d for d in val_data if d["label"] == 0]


# ============================================================================
# LOAD ENCODER
# ============================================================================

tokenizer = AutoTokenizer.from_pretrained(
    config.FINETUNED_MODEL_PATH,
    trust_remote_code=config.TRUST_REMOTE_CODE
)

encoder = AutoModel.from_pretrained(
    config.FINETUNED_MODEL_PATH,
    trust_remote_code=config.TRUST_REMOTE_CODE
).to(device)

encoder.eval()

for p in encoder.parameters():
    p.requires_grad=False


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(data,batch_size=32):

    embeddings=[]

    with torch.no_grad():

        for i in tqdm(range(0,len(data),batch_size)):

            batch=data[i:i+batch_size]
            texts=[d["sequence"] for d in batch]

            inputs = tokenizer(
                texts,
                max_length=config.MAX_LENGTH,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )

            inputs={k:v.to(device) for k,v in inputs.items()}

            outputs=encoder(**inputs)

            cls=outputs.last_hidden_state[:,0,:]

            embeddings.append(cls.cpu().numpy())

    return np.vstack(embeddings)


train_emb = extract_embeddings(train_benign)
val_emb   = extract_embeddings(val_benign)
test_emb  = extract_embeddings(test_data)


# ============================================================================
# DATASET
# ============================================================================

class EmbeddingDataset(Dataset):

    def __init__(self,x):
        self.x=torch.FloatTensor(x)

    def __len__(self):
        return len(self.x)

    def __getitem__(self,i):
        return self.x[i]


# ============================================================================
# AUTOENCODER
# ============================================================================

class Autoencoder(nn.Module):

    def __init__(self):

        super().__init__()

        encoder=[]
        in_dim=config.EMBEDDING_DIM

        for h in config.HIDDEN_DIMS:
            encoder+= [
                nn.Linear(in_dim,h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT)
            ]
            in_dim=h

        encoder.append(nn.Linear(in_dim,config.LATENT_DIM))

        decoder=[]
        in_dim=config.LATENT_DIM

        for h in reversed(config.HIDDEN_DIMS):
            decoder+= [
                nn.Linear(in_dim,h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(config.DROPOUT)
            ]
            in_dim=h

        decoder.append(nn.Linear(in_dim,config.EMBEDDING_DIM))

        self.encoder=nn.Sequential(*encoder)
        self.decoder=nn.Sequential(*decoder)

    def forward(self,x):

        z=self.encoder(x)
        x_hat=self.decoder(z)

        return x_hat,z









def run():
    train_loader = DataLoader(
        EmbeddingDataset(train_emb),
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        EmbeddingDataset(val_emb),
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = Autoencoder().to(device)


    # ============================================================================
    # TRAINING
    # ============================================================================

    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5
    )


    def train_epoch():

        model.train()
        total=0

        for x in train_loader:

            x=x.to(device)

            recon,_=model(x)
            loss=criterion(recon,x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total+=loss.item()

        return total/len(train_loader)


    def validate():

        model.eval()
        total=0

        with torch.no_grad():

            for x in val_loader:

                x=x.to(device)

                recon,_=model(x)
                loss=criterion(recon,x)

                total+=loss.item()

        return total/len(val_loader)



    best_val=float("inf")
    patience=0
    best_state=None

    history={"train":[],"val":[]}

    for epoch in range(config.EPOCHS):

        train_loss=train_epoch()
        val_loss=validate()

        scheduler.step(val_loss)

        history["train"].append(train_loss)
        history["val"].append(val_loss)

        print(f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"train {train_loss:.6f} | val {val_loss:.6f}")

        if val_loss < best_val - config.MIN_DELTA:

            best_val=val_loss
            patience=0
            best_state=copy.deepcopy(model.state_dict())

        else:

            patience+=1

            if patience >= config.PATIENCE:
                print("Early stopping")
                break


    if best_state:
        model.load_state_dict(best_state)

    # ============================================================================
    # SAVE MODEL
    # ============================================================================

    save_path = config.MODEL_DIR / "autoencoder.pth"

    torch.save(
        {
            "model_state_dict":model.state_dict(),
            "config":{
                "input_dim":config.EMBEDDING_DIM,
                "hidden_dims":config.HIDDEN_DIMS,
                "latent_dim":config.LATENT_DIM,
                "dropout":config.DROPOUT
            },
            "history":history,
            "best_val_loss":best_val
        },
        save_path
    )

    print("Autoencoder saved:",save_path)


if __name__ == "__main__":
    run()
