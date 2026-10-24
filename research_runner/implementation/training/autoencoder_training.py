"""Train benign-only AEs, select/calibrate on validation, then optionally test."""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
from pathlib import Path
import time

from research_runner.progress import emit, iterable


def extract_embeddings(encoder_dir, rows, output, device="cuda:0"):
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    from research_runner.implementation.data.data_preprocess import sha256, write_json
    from research_runner.implementation.training.gate_common import directory_hash

    output = Path(output)
    settings = json.loads((Path(encoder_dir) / "training_config.json").read_text())
    fingerprint = directory_hash(encoder_dir)
    import hashlib
    content_hash = hashlib.sha256(json.dumps([(r["id"], r["sequence"]) for r in rows], ensure_ascii=False).encode()).hexdigest()
    identity = {"encoder_sha256": fingerprint, "input_sha256": content_hash,
                "max_length": settings["max_length"], "pooling": "cls", "precision": "bf16", "rows": len(rows)}
    metadata = output.with_suffix(".json")
    if output.exists() and metadata.exists():
        cached = json.loads(metadata.read_text())
        if cached.get("identity") == identity and cached.get("array_sha256") == sha256(output):
            array = np.load(output)
            if array.shape == (len(rows), 768):
                emit('ae.embeddings', len(rows), len(rows), unit='prefixes', status='completed',
                     message=f'Compatible cache: {output.name}', force=True)
                return array
    emit('ae.encoder_load', message=f'Loading encoder for {output.name}.', force=True)
    tokenizer = AutoTokenizer.from_pretrained(encoder_dir, local_files_only=True)
    encoder = AutoModel.from_pretrained(encoder_dir, local_files_only=True,
                                        attn_implementation="sdpa", reference_compile=False).to(device).eval()
    encoder.requires_grad_(False)
    values = []
    start = time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(iterable(rows, f'ae.embeddings.{output.name}', unit='prefixes')):
            batch = tokenizer(row["sequence"], truncation=True, max_length=settings["max_length"],
                              padding=True, return_tensors="pt")
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                cls = encoder(**batch).last_hidden_state[:, 0, :]
            values.append(cls.float().cpu().numpy())
            if (i+1) % 250 == 0:
                print(f"EMBEDDINGS {output.name}: {i+1}/{len(rows)} seconds={time.perf_counter()-start:.1f}", flush=True)
    array = np.concatenate(values)
    if not np.isfinite(array).all():
        raise ValueError("Nonfinite encoder embeddings")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, array)
    write_json(metadata, {"identity": identity, "array_sha256": sha256(output),
                          "seconds": time.perf_counter()-start})
    del encoder
    torch.cuda.empty_cache()
    return array


def reconstruction_errors(model, embeddings, device):
    import torch
    import numpy as np
    model.eval()
    values = []
    with torch.no_grad():
        for i in range(0, len(embeddings), 128):
            batch = torch.as_tensor(embeddings[i:i+128], dtype=torch.float32, device=device)
            reconstructed, _ = model(batch)
            values.append(torch.mean((reconstructed-batch)**2, dim=1).cpu().numpy())
    return np.concatenate(values)


def run(*, encoder_dir=None, output_dir=None, seed=42, recall_floor=0.98,
        search=False, evaluate_test=False):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from research_runner.implementation.training.settings import AutoEncoderConfig
    from research_runner.implementation.data.data_preprocess import ROOT, read_jsonl, sha256, write_json, write_jsonl
    from research_runner.implementation.data.prepare_training import load_prepared
    from research_runner.implementation.detectors.gate import AutoEncoder
    from research_runner.implementation.training.gate_common import calibrate, directory_hash, metrics, provenance, timestamp

    config = AutoEncoderConfig()
    if not 0 <= recall_floor <= 1:
        raise ValueError("recall_floor must be between zero and one")
    encoder_dir = Path(encoder_dir or config.FINETUNED_MODEL_PATH)
    output = Path(output_dir or config.MODEL_DIR)
    if (output / "selection_frozen.json").exists():
        raise ValueError(f"Refusing to overwrite a selected AE run: {output}")
    output.mkdir(parents=True, exist_ok=True)
    encoder_settings = json.loads((encoder_dir / "training_config.json").read_text())
    if encoder_settings["max_length"] != config.MAX_LENGTH:
        raise ValueError("Encoder/AE max lengths differ; select the matching trained encoder")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    train_rows = load_prepared("autoencoder", "train")
    benign_val_rows = load_prepared("autoencoder", "val")
    val_rows = load_prepared("encoder", "val")
    # Calibration and hyperparameter selection never load correlated or test data.
    train_emb = extract_embeddings(encoder_dir, train_rows, output / "embeddings/train_benign.npy")
    val_emb = extract_embeddings(encoder_dir, val_rows, output / "embeddings/validation.npy")
    position = {r["id"]: i for i, r in enumerate(val_rows)}
    val_benign_emb = val_emb[[position[r["id"]] for r in benign_val_rows]]
    labels = np.array([r["label"] for r in val_rows])
    grid = list(itertools.product(config.SEARCH_LEARNING_RATES, config.SEARCH_LATENT_DIMS)) if search else [(config.LEARNING_RATE, config.LATENT_DIM)]
    settings = {"encoder_path": str(encoder_dir), "encoder_sha256": directory_hash(encoder_dir),
                "max_length": config.MAX_LENGTH, "embedding_dim": 768, "pooling": "cls",
                "embedding_precision": "bf16", "embedding_batch_size": 1,
                "hidden_dims": list(config.HIDDEN_DIMS), "dropout": config.DROPOUT,
                "epochs": config.EPOCHS, "batch_size": config.BATCH_SIZE,
                "weight_decay": config.WEIGHT_DECAY, "patience": config.PATIENCE,
                "min_delta": config.MIN_DELTA, "seed": seed, "optimizer": "Adam",
                "scheduler": {"name": "ReduceLROnPlateau", "factor": 0.5, "patience": 5},
                "train_rows": len(train_rows), "benign_validation_rows": len(benign_val_rows),
                "calibration_rows": len(val_rows), "calibration_scope": "prepared_noncorrelated_validation",
                "recall_floor": recall_floor, "grid": [{"learning_rate": lr, "latent_dim": latent} for lr, latent in grid],
                "source_sha256": sha256(Path(__file__)), "provenance": provenance(), "started_at_utc": timestamp()}
    write_json(output / "run_config.json", settings)
    write_json(output / "sample_ids.json", {"train": [r["id"] for r in train_rows],
                                           "benign_validation": [r["id"] for r in benign_val_rows],
                                           "calibration": [r["id"] for r in val_rows]})
    candidates = []
    for lr, latent in iterable(grid, 'ae.candidates', unit='candidates'):
        phase = f'ae.lr_{lr:g}.latent_{latent}'
        emit(phase, 0, config.EPOCHS, unit='epochs', message=f'Seed={seed}', force=True)
        torch.manual_seed(seed); np.random.seed(seed)
        candidate_dir = output / f"lr_{lr:g}_latent_{latent}"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        model = AutoEncoder(768, config.HIDDEN_DIMS, latent, config.DROPOUT).to("cuda:0")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        loader = DataLoader(TensorDataset(torch.as_tensor(train_emb)), batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=0, drop_last=len(train_emb) % config.BATCH_SIZE == 1,
                            generator=torch.Generator().manual_seed(seed))
        best_loss, stale, best_state, history, best_epoch = float("inf"), 0, None, [], None
        for epoch in range(config.EPOCHS):
            model.train(); total, count = 0.0, 0
            for (batch,) in iterable(loader, f'{phase}.epoch_{epoch+1}', unit='batches'):
                batch = batch.to("cuda:0")
                optimizer.zero_grad(set_to_none=True)
                reconstructed, _ = model(batch)
                loss = ((reconstructed-batch)**2).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite AE training loss")
                loss.backward(); optimizer.step()
                total += float(loss.detach())*len(batch); count += len(batch)
            val_loss = float(reconstruction_errors(model, val_benign_emb, "cuda:0").mean())
            scheduler.step(val_loss)
            history.append({"epoch": epoch+1, "train_loss": total/count, "val_loss": val_loss,
                            "learning_rate": optimizer.param_groups[0]["lr"]})
            print(f"AE lr={lr:g} latent={latent} epoch={epoch+1} train={total/count:.7f} val={val_loss:.7f}", flush=True)
            emit(phase, epoch+1, config.EPOCHS, unit='epochs',
                 message=f'Train loss={total/count:.7f}; validation loss={val_loss:.7f}', force=True)
            if val_loss < best_loss-config.MIN_DELTA:
                best_loss, stale, best_state = val_loss, 0, copy.deepcopy(model.state_dict())
                best_epoch = epoch+1
            else:
                stale += 1
                if stale >= config.PATIENCE:
                    break
        model.load_state_dict(best_state)
        emit(phase, epoch+1, config.EPOCHS, unit='epochs', status='completed',
             message=f'Training ended; selected epoch={best_epoch}. Calibrating on validation.', force=True)
        errors = reconstruction_errors(model, val_emb, "cuda:0")
        calibration = calibrate(labels, errors, recall_floor)
        checkpoint = {"model_state_dict": best_state,
                      "config": {"input_dim": 768, "hidden_dims": list(config.HIDDEN_DIMS),
                                 "latent_dim": latent, "dropout": config.DROPOUT},
                      "encoder_sha256": settings["encoder_sha256"], "training_config": settings,
                      "learning_rate": lr, "best_val_loss": best_loss, "history": history,
                      "best_epoch": best_epoch, "optimizer_parameters": optimizer.defaults,
                      "calibration": calibration}
        checkpoint_path = candidate_dir / "autoencoder.pth"
        torch.save(checkpoint, checkpoint_path)
        candidate = {"learning_rate": lr, "latent_dim": latent, "checkpoint": str(checkpoint_path),
                     "checkpoint_sha256": sha256(checkpoint_path), "best_val_loss": best_loss,
                     "best_epoch": best_epoch, "epochs_completed": len(history), "calibration": calibration}
        write_json(candidate_dir / "training_config.json", {**settings, **candidate, "history": history,
                   "optimizer_parameters": optimizer.defaults, "architecture": str(model),
                   "shuffle_seed": seed, "drop_last": loader.drop_last})
        np.save(candidate_dir / "validation_errors.npy", errors)
        candidates.append(candidate)
        del model
    best = max(candidates, key=lambda c: (c["calibration"]["metrics"]["f1"],
                                         -c["calibration"]["metrics"]["fpr"], -c["best_val_loss"]))
    selection = {"selection_source": "validation_only", "scope": "best_of_declared_grid_not_global_optimum",
                 "frozen_at_utc": timestamp(), "settings": settings, "best": best, "candidates": candidates}
    write_json(output / "selection_frozen.json", selection)
    print(f"AE SELECTION FROZEN: {best}", flush=True)
    if evaluate_test:
        # This is deliberately after selection has been sealed on disk.
        test_path = ROOT / "data/processed/test_processed.jsonl"
        test_rows = read_jsonl(test_path)
        test_emb = extract_embeddings(encoder_dir, test_rows, output / "embeddings/test.npy")
        checkpoint = torch.load(best["checkpoint"], map_location="cpu", weights_only=False)
        model = AutoEncoder(**checkpoint["config"]).to("cuda:0")
        model.load_state_dict(checkpoint["model_state_dict"])
        test_errors = reconstruction_errors(model, test_emb, "cuda:0")
        threshold = best["calibration"]["threshold"]
        test_labels = np.array([r["label"] for r in test_rows])
        verified = np.array([not r.get("annotation_requires_review", True) for r in test_rows])
        report = {"threshold_frozen_from_validation": threshold, "test_sha256": sha256(test_path),
                  "selection_sha256": sha256(output / "selection_frozen.json"),
                  "all_operational_labels": metrics(test_labels, test_errors, threshold),
                  "excluding_unresolved_annotations": metrics(test_labels[verified], test_errors[verified], threshold),
                  "unresolved_annotation_rows": int((~verified).sum()),
                  "per_scenario": {kind: metrics(test_labels[mask], test_errors[mask], threshold)
                                   for kind in ("benign", "prompt", "rag", "tool", "correlated")
                                   if np.any(mask := np.array([(r.get("attack_type") or "benign") == kind for r in test_rows]))}}
        write_json(output / "test_report.json", report)
        write_jsonl(output / "test_predictions.jsonl", ({"id": r["id"], "label": r["label"],
                    "annotation_requires_review": r["annotation_requires_review"],
                    "pred": int(error > threshold), "error": float(error)} for r, error in zip(test_rows, test_errors)))
        print(f"FROZEN TEST REPORT {report}", flush=True)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default=os.environ.get("CUDA_DEVICE_TRAIN", os.environ.get("CUDA_DEVICE_ML", "0")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall-floor", type=float, default=0.98)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    os.environ["CUDA_DEVICE_ML"] = args.device
    from research_runner.implementation.training.gate_runtime import gpu_preflight
    gpu_preflight(args.device)
    if args.output_dir is None:
        from datetime import datetime, timezone
        from research_runner.implementation.data.data_preprocess import ROOT
        args.output_dir = ROOT / "results/research_runner/training" / datetime.now(timezone.utc).strftime("ae_%Y%m%d_%H%M%S")
    run(encoder_dir=args.encoder, output_dir=args.output_dir, seed=args.seed,
        recall_floor=args.recall_floor, search=args.search, evaluate_test=args.evaluate_test)


if __name__ == "__main__":
    main()
