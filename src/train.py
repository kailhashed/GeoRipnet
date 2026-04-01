"""
train.py
Training script for the main GeoRipNet model and its ablations.
"""
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pywt

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR, NODES,
    N_NODES, LOOKBACK_WINDOW,
    BATCH_SIZE, LR, EPOCHS, PATIENCE,
    DEVICE, WEIGHT_DECAY
)
from model import GeoRipNet

try:
    from baselines import sliding_windows, metrics
except ImportError:
    raise ImportError("baselines.py must be in the same directory as train.py")


def apply_wavelet_smoothing(data_array, wavelet='db2', level=2):
    smoothed = np.zeros_like(data_array)
    for i in range(data_array.shape[1]):
        signal = data_array[:, i]
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs[-1] = np.zeros_like(coeffs[-1])  # Zero out highest freq noise
        smoothed[:, i] = pywt.waverec(coeffs, wavelet)[:len(signal)]
    return smoothed.astype(np.float32)

class OilDataset(Dataset):
    """Dataset for GeoRipNet returning prices, gdelt, adjacency, and targets."""
    def __init__(self, split_dir, lookback=LOOKBACK_WINDOW, price_mean=None, price_std=None, horizon=1, stride=1):
        split_dir = Path(split_dir)
        prices_df = pd.read_parquet(split_dir / "prices.parquet")
        prices_df.index = pd.to_datetime(prices_df.index)
        prices_df = prices_df.sort_index()
        
        raw_prices = prices_df.values.astype(np.float32)

        # Normalise inputs — mean/std from training set
        if price_mean is None:
            self.price_mean = raw_prices.mean(axis=0).astype(np.float32)
            self.price_std  = raw_prices.std(axis=0).clip(min=1.0).astype(np.float32)
        else:
            self.price_mean = price_mean
            self.price_std  = price_std

        # Apply Wavelet Smoothing to inputs before normalization
        smoothed_prices = apply_wavelet_smoothing(raw_prices, wavelet='db2', level=2)
        prices_norm = (smoothed_prices - self.price_mean) / self.price_std

        self.X_prices = []
        y_ret_list = []
        y_dir_list = []
        y_raw_list = []
        last_norm_list = []
        target_dates_seq = []
        last_obs_dates = []

        for i in range(lookback, len(prices_norm) - horizon + 1, stride):
            t = i - 1
            self.X_prices.append(prices_norm[i - lookback : i])
            last_norm = prices_norm[t]
            last_norm_list.append(last_norm)
            
            y_ret_h = []
            y_dir_h = []
            y_raw_h = []
            for d in range(1, horizon + 1):
                ret = prices_norm[t + d] - last_norm
                y_ret_h.append(ret)
                y_dir_h.append(ret > 0)
                y_raw_h.append(raw_prices[t + d])
                
            y_ret_list.append(y_ret_h)
            y_dir_list.append(y_dir_h)
            y_raw_list.append(y_raw_h)
            
            target_dates_seq.append(prices_df.index[t + 1 : t + 1 + horizon])
            last_obs_dates.append(prices_df.index[t])

        self.X_prices = np.array(self.X_prices, dtype=np.float32)
        self.y_returns = np.array(y_ret_list, dtype=np.float32)
        self.y_dir     = np.array(y_dir_list, dtype=np.float32)
        self.y_raw     = np.array(y_raw_list, dtype=np.float32)
        self.last_norm = np.array(last_norm_list, dtype=np.float32)

        self.target_dates_seq = target_dates_seq
        self.last_obs_dates = pd.DatetimeIndex(last_obs_dates)
        self.valid_indices  = np.arange(len(self.X_prices))
        
        adj_df = pd.read_parquet(split_dir / "adjacency.parquet")
        adj_cols = [f"col_{r}_{c}" for r in range(N_NODES) for c in range(N_NODES)]
        adjs = []
        for d in self.last_obs_dates:
            period = d.year * 100 + d.month
            row = adj_df[adj_df["period"] == period]
            if row.empty:
                row = adj_df[adj_df["period"] <= period].iloc[-1:]
            if row.empty:
                adjs.append(np.eye(N_NODES, dtype=np.float32) / N_NODES)
            else:
                adjs.append(row[adj_cols].values.reshape(N_NODES, N_NODES))
        self.adjs = np.array(adjs, dtype=np.float32)
        
        # Load GDELT from split folder — vectorized, no iterrows
        gdelt_raw = pd.read_parquet(split_dir / "gdelt.parquet")
        gdelt_raw["date"] = pd.to_datetime(gdelt_raw["date"])
        gdelt_raw = gdelt_raw[gdelt_raw["from_node"] != gdelt_raw["to_node"]].copy()
        gdelt_dates = sorted(gdelt_raw["date"].unique())
        date_to_idx_g = {d: i for i, d in enumerate(gdelt_dates)}
        row_idx_g = gdelt_raw["date"].map(date_to_idx_g).values
        fi_g = gdelt_raw["from_node"].values.astype(int)
        ti_g = gdelt_raw["to_node"].values.astype(int)
        gdelt_arr = np.zeros((len(gdelt_dates), 3 * N_NODES * N_NODES), dtype=np.float32)
        for ch_i, ch in enumerate(["GoldsteinScale", "AvgTone", "NumMentions"]):
            flat_idx = ch_i * N_NODES * N_NODES + fi_g * N_NODES + ti_g
            gdelt_arr[row_idx_g, flat_idx] = gdelt_raw[ch].values.astype(np.float32)
        gdelt_wide = pd.DataFrame(gdelt_arr, index=pd.DatetimeIndex(gdelt_dates))
        gdelt_flat_df = gdelt_wide.reindex(prices_df.index, method="ffill").fillna(0.0)

        gdelt_flat = gdelt_flat_df.values
        gdelt_tensor = gdelt_flat.reshape(-1, 3, N_NODES, N_NODES).transpose(0, 2, 3, 1)
        
        gdelt_seqs = []
        for i in range(lookback, len(prices_norm) - horizon + 1, stride):
            gdelt_seqs.append(gdelt_tensor[i - lookback:i])
        self.gdelt = np.array(gdelt_seqs, dtype=np.float32)

    def __len__(self):
        return len(self.X_prices)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X_prices[i]),       # [k, 5] normalised price window
            torch.tensor(self.gdelt[i]),           # [k, 5, 5, 3] GDELT tensor
            torch.tensor(self.adjs[i]),            # [5, 5] adjacency
            torch.tensor(self.y_returns[i]),       # [5]   log return target
            torch.tensor(self.y_dir[i]),           # [5]   direction label (0/1)
        )


def evaluate_and_save(model_name, y_true, y_pred, test_dates_window, lookback):
    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # y_true and y_pred are already flattened: [N_total_steps, 5]
    # test_dates_window is the flattened DatetimeIndex: [N_total_steps]
    res_overall = metrics(y_true, y_pred)
    calm_mask = test_dates_window < pd.to_datetime("2022-02-24")
    crisis_mask = test_dates_window >= pd.to_datetime("2022-02-24")
    
    res_calm = metrics(y_true[calm_mask], y_pred[calm_mask]) if calm_mask.sum() > 0 else {}
    res_crisis = metrics(y_true[crisis_mask], y_pred[crisis_mask]) if crisis_mask.sum() > 0 else {}
        
    full_res = {
        "overall": res_overall,
        "calm": res_calm,
        "crisis": res_crisis
    }
    
    df_pred = pd.DataFrame(y_pred, index=test_dates_window, columns=NODES)
    df_pred.to_csv(out_dir / f"predictions_k{lookback}.csv")
        
    with open(out_dir / f"metrics_k{lookback}.json", "w") as f:
        json.dump(full_res, f, indent=2)
        
    return full_res


def run_training_ablation(lookback: int, ablation: str, horizon: int):
    print("=" * 60)
    print(f"Training GeoRipNet [{ablation}] | lookback={lookback} | horizon={horizon} | device={DEVICE}")
    print("=" * 60)

    # Train overlapping (stride 1)
    tr_ds  = OilDataset(TRAIN_DIR, lookback, horizon=horizon, stride=1)
    val_ds = OilDataset(VAL_DIR, lookback, price_mean=tr_ds.price_mean, price_std=tr_ds.price_std, horizon=horizon, stride=1)
    # Test strided (non-overlapping)
    te_ds  = OilDataset(TEST_DIR, lookback, price_mean=tr_ds.price_mean, price_std=tr_ds.price_std, horizon=horizon, stride=horizon)

    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    te_dl  = DataLoader(te_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = GeoRipNet(lookback=lookback, ablation=ablation, horizon=horizon).to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    mse_crit = nn.MSELoss()
    bce_crit = nn.BCEWithLogitsLoss()
    LAMBDA_DIR = 3.0   # aggressively increased direction loss weight to optimize DA

    best_loss    = float("inf")
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        tr_losses = []
        for X, gdelt_batch, adj, y_ret, y_dir in tr_dl:
            X         = X.to(DEVICE)
            gdelt_batch = gdelt_batch.to(DEVICE)
            adj       = adj.to(DEVICE)
            y_ret     = y_ret.to(DEVICE)
            y_dir     = y_dir.to(DEVICE)
            opt.zero_grad()
            log_ret_hat, dir_hat = model(X, gdelt_batch, adj)
            loss = mse_crit(log_ret_hat, y_ret) + LAMBDA_DIR * bce_crit(dir_hat, y_dir)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            tr_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, gdelt_batch, adj, y_ret, y_dir in val_dl:
                X         = X.to(DEVICE)
                gdelt_batch = gdelt_batch.to(DEVICE)
                adj       = adj.to(DEVICE)
                y_ret     = y_ret.to(DEVICE)
                y_dir     = y_dir.to(DEVICE)
                log_ret_hat, dir_hat = model(X, gdelt_batch, adj)
                loss = mse_crit(log_ret_hat, y_ret) + LAMBDA_DIR * bce_crit(dir_hat, y_dir)
                val_losses.append(loss.item())

        tr_loss  = float(np.mean(tr_losses))
        val_loss = float(np.mean(val_losses))
        sch.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {
                "lookback":    lookback,
                "price_mean":  tr_ds.price_mean.tolist(),
                "price_std":   tr_ds.price_std.tolist(),
                "model_state": {k: v.clone() for k, v in model.state_dict().items()}
            }
            patience_cnt = 0
            torch.save(best_state, CHECKPOINT_DIR / f"georipnet_{ablation}_k{lookback}_best.pt")
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  Early stop epoch {epoch} | best val {best_loss:.6f}")
                break

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train: {tr_loss:.6f} | Val: {val_loss:.6f} | Pat: {patience_cnt}")

    if best_state is not None:
        model.load_state_dict(best_state["model_state"])

    # ── Evaluation ────────────────────────────────────────────────────────────
    print("\nEvaluating on Test Set...")
    model.eval()
    log_ret_preds = []
    with torch.no_grad():
        for X, gdelt_batch, adj, y_ret, y_dir in te_dl:
            X           = X.to(DEVICE)
            gdelt_batch = gdelt_batch.to(DEVICE)
            adj         = adj.to(DEVICE)
            log_ret_hat, _ = model(X, gdelt_batch, adj)
            log_ret_preds.append(log_ret_hat.cpu().numpy())

    # Flatten the independent sequence chunks to form a cohesive timeline
    log_ret_preds = np.vstack(log_ret_preds)   # [N_chunks, horizon, 5]
    log_ret_flat  = log_ret_preds.reshape(-1, N_NODES) # [N_total_steps, 5]
    
    # Reconstruct real prices: P_hat_norm = P_t + Δ_pred → denormalise
    preds_norm  = np.expand_dims(te_ds.last_norm, 1) + log_ret_preds
    preds_flat  = (preds_norm * tr_ds.price_std + tr_ds.price_mean).reshape(-1, N_NODES)
    
    y_true_real = te_ds.y_raw                                      
    y_true_flat = y_true_real.reshape(-1, N_NODES)

    y_true_ret = te_ds.y_returns.reshape(-1, N_NODES)
    da_per_node = (np.sign(log_ret_flat) == np.sign(y_true_ret)).mean(axis=0)
    print(f"  Directional Accuracy per node: " +
          " | ".join(f"{n}={100*da:.1f}%" for n, da in zip(NODES, da_per_node)))

    flat_dates = np.concatenate(te_ds.target_dates_seq)
    flat_dates_idx = pd.DatetimeIndex(flat_dates)

    res = evaluate_and_save(
        f"GeoRipNet_{ablation}_h{horizon}", y_true_flat, preds_flat,
        flat_dates_idx, lookback
    )
    
    print(f"\nTest Metrics (price space) for Horizon={horizon}:")
    for n in NODES:
        print(f"    {n:8s} | RMSE={res['overall']['RMSE'][n]:.3f} | R²={res['overall']['R2'][n]:.4f} | DA={res['overall']['DA'][n]:.1f}%")
    print(f"    {'MEAN':8s} | RMSE={res['overall']['RMSE_mean']:.3f} | R²={res['overall']['R2_mean']:.4f} | DA={res['overall']['DA_mean']:.1f}%")
    return res


def run_all(lookback: int):
    # Standard complete run
    from config import HORIZONS
    for h in HORIZONS:
        run_training_ablation(lookback, "A5", h)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback", type=int, default=LOOKBACK_WINDOW)
    parser.add_argument("--ablation", type=str, default="A5")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    
    if args.all:
        run_all(args.lookback)
    else:
        run_training_ablation(args.lookback, args.ablation)
