"""
train.py
Training loop for GeoRipNet.

Usage:
  python train.py --lookback 20
  python train.py --lookback 10 --epochs 150
"""
import sys
import argparse, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END,
    BATCH_SIZE, LR, EPOCHS, PATIENCE, LOOKBACK_WINDOW, N_NODES
)
from model import GeoRipNet


# ── Dataset ───────────────────────────────────────────────────────────────────

class OilDataset(Dataset):
    def __init__(self, samples: pd.DataFrame, split: str,
                 train_start=TRAIN_START, train_end=TRAIN_END,
                 val_start=VAL_START, val_end=VAL_END):
        df = samples.copy()
        df["date"] = pd.to_datetime(df["date"])
        if split == "train":
            df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
        elif split == "val":
            df = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
        else:  # test
            from config import TEST_START
            df = df[df["date"] >= TEST_START]
        self.data = df.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        prices   = torch.tensor(row["prices_window"], dtype=torch.float32)   # [k, 5]
        gdelt    = torch.tensor(row["gdelt_tensor"], dtype=torch.float32).view(5, 5, 3)
        adj      = torch.tensor(row["adjacency"], dtype=torch.float32)       # [5, 5]
        target   = torch.tensor(row["target"], dtype=torch.float32)          # [5]
        return prices, gdelt, adj, target


# ── Training ──────────────────────────────────────────────────────────────────

def train(lookback: int = LOOKBACK_WINDOW, epochs: int = EPOCHS):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    samples = pd.read_parquet(DATA_DIR / f"dataset_k{lookback}.parquet")

    train_ds = OilDataset(samples, "train")
    val_ds   = OilDataset(samples, "val")
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    model     = GeoRipNet(lookback=lookback).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for prices, gdelt, adj, target in train_dl:
            prices, gdelt, adj, target = (
                prices.to(device), gdelt.to(device),
                adj.to(device), target.to(device)
            )
            optimizer.zero_grad()
            pred = model(prices, gdelt, adj)
            loss = criterion(pred, target)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # Validate
        model.eval()
        val_losses = []
        with torch.no_grad():
            for prices, gdelt, adj, target in val_dl:
                prices, gdelt, adj, target = (
                    prices.to(device), gdelt.to(device),
                    adj.to(device), target.to(device)
                )
                pred = model(prices, gdelt, adj)
                val_losses.append(criterion(pred, target).item())

        train_loss = np.mean(train_losses)
        val_loss   = np.mean(val_losses)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        # Early stopping + checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt = CHECKPOINT_DIR / f"georipnet_k{lookback}_best.pt"
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "val_loss": val_loss, "lookback": lookback}, ckpt)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

    # Save history
    json.dump(history, open(RESULTS_DIR / f"history_k{lookback}.json", "w"))
    print(f"Best val loss: {best_val_loss:.4f} | Checkpoint: {ckpt}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=LOOKBACK_WINDOW)
    parser.add_argument("--epochs",   type=int, default=EPOCHS)
    args = parser.parse_args()
    train(lookback=args.lookback, epochs=args.epochs)
