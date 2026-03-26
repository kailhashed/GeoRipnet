"""
evaluate.py
Test-set evaluation, ablation experiments, and figure generation.

Usage:
  python evaluate.py --lookback 20
  python evaluate.py --lookback 20 --ablation no_gdelt
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from config import (
    DATA_DIR, CHECKPOINT_DIR, RESULTS_DIR,
    N_NODES, NODES, TEST_START
)
from model import GeoRipNet
from train import OilDataset


ABLATIONS = ["full", "no_gdelt", "no_dynamic_graph", "gcn", "lstm", "single_benchmark"]


# ── Metrics ───────────────────────────────────────────────────────────────────

def mae(y_true, y_pred):   return np.abs(y_true - y_pred).mean(axis=0)
def rmse(y_true, y_pred):  return np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
def mape(y_true, y_pred):  return (np.abs((y_true - y_pred) / (y_true + 1e-8))).mean(axis=0) * 100


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(lookback: int = 20, ablation: str = "full"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples  = pd.read_parquet(DATA_DIR / f"dataset_k{lookback}.parquet")
    test_ds  = OilDataset(samples, "test")
    test_dl  = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)
    print(f"Test samples: {len(test_ds)}")

    # Load checkpoint
    ckpt_path = CHECKPOINT_DIR / f"georipnet_k{lookback}_best.pt"
    ckpt      = torch.load(ckpt_path, map_location=device)
    model     = GeoRipNet(lookback=lookback).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_preds, all_targets, all_dates = [], [], []

    with torch.no_grad():
        for prices, gdelt, adj, target in test_dl:
            prices, gdelt, adj = prices.to(device), gdelt.to(device), adj.to(device)

            if ablation == "no_gdelt":
                # Replace A_dynamic with A_static (skip gating)
                gdelt = torch.zeros_like(gdelt)
            elif ablation == "no_dynamic_graph":
                # Replace A_dynamic with fixed correlation matrix (identity proxy)
                adj = torch.eye(N_NODES).unsqueeze(0).expand(adj.shape[0], -1, -1).to(device)

            pred = model(prices, gdelt, adj)
            all_preds.append(pred.cpu().numpy())
            all_targets.append(target.numpy())

    preds   = np.vstack(all_preds)
    targets = np.vstack(all_targets)

    results = {
        "ablation": ablation,
        "lookback": lookback,
        "MAE":  dict(zip(NODES, mae(targets, preds).tolist())),
        "RMSE": dict(zip(NODES, rmse(targets, preds).tolist())),
        "MAPE": dict(zip(NODES, mape(targets, preds).tolist())),
        "MAE_mean":  float(mae(targets, preds).mean()),
        "RMSE_mean": float(rmse(targets, preds).mean()),
    }

    out = RESULTS_DIR / f"eval_{ablation}_k{lookback}.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"\n=== {ablation.upper()} k={lookback} ===")
    for metric in ["MAE", "RMSE", "MAPE"]:
        vals = results[metric]
        print(f"  {metric}: " + " | ".join(f"{n}={v:.3f}" for n, v in vals.items()))

    return preds, targets, results


# ── Paper Figures ─────────────────────────────────────────────────────────────

def figure_gate_urals_india(model, test_loader, device):
    """Figure 1: Gate[3,4] (Urals→India edge) over test period."""
    gates = []
    model.eval()
    with torch.no_grad():
        for prices, gdelt, adj, _ in test_loader:
            gdelt, adj = gdelt.to(device), adj.to(device)
            g = torch.sigmoid(model.edge_gating.W_g(gdelt).squeeze(-1))
            gates.append(g[:, 3, 4].cpu().numpy())

    gates = np.concatenate(gates)
    plt.figure(figsize=(12, 4))
    plt.plot(gates, linewidth=0.8)
    plt.axvline(x=0, color='red', linestyle='--', label='Feb 24 2022 (invasion)')
    plt.title("Gate[Urals→India] over test period (2022–present)")
    plt.ylabel("Gate value (0=suppressed, 1=full capacity)")
    plt.xlabel("Days since test start")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig1_gate_urals_india.png", dpi=150)
    plt.close()
    print("Saved: fig1_gate_urals_india.png")


def figure_prediction_vs_actual(preds, targets):
    """Figure 3: Predicted vs actual for all 5 benchmarks."""
    fig, axes = plt.subplots(N_NODES, 1, figsize=(14, 12), sharex=True)
    for i, (ax, name) in enumerate(zip(axes, NODES)):
        ax.plot(targets[:, i], label="Actual",    linewidth=0.8, color="black")
        ax.plot(preds[:, i],   label="Predicted", linewidth=0.8, color="steelblue", alpha=0.8)
        ax.set_ylabel(f"{name} ($/bbl)")
        ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Days in test period (2022–present)")
    fig.suptitle("GeoRipNet: Predicted vs Actual — All 5 Benchmarks")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "fig3_pred_vs_actual.png", dpi=150)
    plt.close()
    print("Saved: fig3_pred_vs_actual.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ablation", type=str, default="full",
                        choices=ABLATIONS)
    args = parser.parse_args()
    preds, targets, results = evaluate(args.lookback, args.ablation)
