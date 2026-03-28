"""
baselines.py
Comparison models for GeoRipNet evaluation.

All baselines expose:
    fit(train_prices, ...)
    predict(test_prices, ...) -> np.ndarray [N_test, 5]

Groups:
  Group 1 — Naive Baselines:
    1a. RandomWalk        — y_hat_{t+1} = y_t
    1b. HistoricalMean    — rolling mean of last k days
    1c. ARIMABaseline     — per-benchmark ARIMA(5,1,0)
    1d. ARIMAXBaseline    — ARIMA with GDELT channels as exogenous covariates

  Group 2 — ML Baselines Without Graph Structure:
    2a. XGBoostBaseline   — XGBoost per-benchmark, flattened lookback features
    2b. SVRBaseline       — Support Vector Regression per benchmark
    2c. LSTMBaseline      — 2-layer LSTM, prices only, no GDELT, no graph
    2d. LSTMGDELTBaseline — LSTM with GDELT flat (80-dim input), no graph

  Group 3 — Graph Baselines Without Dynamic Edges:
    3a. GCNStaticBaseline — Standard GCN on fixed Comtrade matrix, no GDELT
    3b. GATStaticBaseline — GAT with fixed A_static, no GDELT gating
                            (renamed from StaticGATBaseline; alias kept)
    3c. DCRNNBaseline     — Diffusion Convolutional RNN (bidirectional)

  Group 4 — Strong Temporal Baselines:
    4a. TFTBaseline       — Temporal Fusion Transformer
    4b. PatchTSTBaseline  — Patch-based time-series Transformer
    4c. NBEATSBaseline    — N-BEATS (trend + seasonality blocks)

  Group 5 — Domain-Specific Prior Work:
    5a. EMDLSTMBaseline        — EMD decomposition + LSTM on IMFs
    5b. SentimentLSTMBaseline  — LSTM with GDELT AvgTone flat sentiment
    5c. GPRLSTMBaseline        — Geopolitical Risk proxy + LSTM

Usage:
    python src/baselines.py --lookback 20
"""
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DATA_DIR, RESULTS_DIR, CHECKPOINT_DIR,
    TRAIN_DIR, VAL_DIR, TEST_DIR,
    NODES, N_NODES, LOOKBACK_WINDOW,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START,
    BATCH_SIZE, LR, EPOCHS, PATIENCE, D_MODEL, N_HEADS_GAT, DROPOUT,
    GDELT_TENSOR_FILE, DEVICE,
)

warnings.filterwarnings("ignore")

# GDELT flat dimension: 5x5x3 channels, diagonal zeroed => 75 non-trivial entries
# (full 5x5x3 = 75, diagonal pairs i==j contribute 15, so 60 off-diag + 15 diag = 75 total)
# We keep all 75 for simplicity (diagonal will be 0).
GDELT_FLAT_DIM = 75   # 5 * 5 * 3


# =============================================================================
# Shared helpers
# =============================================================================

def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, MAPE, R², DA per benchmark + mean."""
    _mae  = np.abs(y_true - y_pred).mean(axis=0)
    _rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    _mape = (np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-8))).mean(axis=0) * 100
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0)
    _r2   = 1 - ss_res / (ss_tot + 1e-8)
    true_diff = np.diff(y_true, axis=0)
    pred_diff = np.diff(y_pred, axis=0)
    _da   = (np.sign(true_diff) == np.sign(pred_diff)).mean(axis=0) * 100
    return {
        "MAE":       dict(zip(NODES, _mae.tolist())),
        "RMSE":      dict(zip(NODES, _rmse.tolist())),
        "MAPE":      dict(zip(NODES, _mape.tolist())),
        "R2":        dict(zip(NODES, _r2.tolist())),
        "DA":        dict(zip(NODES, _da.tolist())),
        "MAE_mean":  float(_mae.mean()),
        "RMSE_mean": float(_rmse.mean()),
        "MAPE_mean": float(_mape.mean()),
        "R2_mean":   float(_r2.mean()),
        "DA_mean":   float(_da.mean()),
    }


def load_prices() -> pd.DataFrame:
    """Load aligned prices — try aligned_prices.parquet, fall back to split dirs."""
    aligned = DATA_DIR / "aligned_prices.parquet"
    if aligned.exists():
        prices = pd.read_parquet(aligned)
        prices.index = pd.to_datetime(prices.index)
        return prices
    # Concatenate from pre-split dirs
    dfs = []
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        df = pd.read_parquet(d / "prices.parquet")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        dfs.append(df)
    return pd.concat(dfs).sort_index()


def sliding_windows(prices: np.ndarray, lookback: int):
    """Return (X [N, lookback, 5], y [N, 5]) pairs from a price array."""
    X, y = [], []
    for i in range(lookback, len(prices)):
        X.append(prices[i - lookback:i])
        y.append(prices[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_gdelt_flat(prices_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Load the GDELT tensor and return a date-indexed DataFrame with 75 columns.

    Columns ordered channel-major: [gs_0_0 ... gs_4_4, at_0_0 ... at_4_4, nm_0_0 ... nm_4_4]
    (75 = 3 channels × 5×5 pairs; diagonal entries are zero).

    Parameters
    ----------
    prices_df : optional reference price DataFrame; used to align dates.

    Returns
    -------
    pd.DataFrame  shape [n_dates, 75], float32, index=DatetimeIndex
    """
    raw = pd.read_parquet(GDELT_TENSOR_FILE)
    raw["date"] = pd.to_datetime(raw["date"])
    raw = raw[raw["from_node"] != raw["to_node"]].copy()

    dates = sorted(raw["date"].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    row_idx = raw["date"].map(date_to_idx).values

    fi = raw["from_node"].values.astype(int)
    ti = raw["to_node"].values.astype(int)

    arr = np.zeros((len(dates), GDELT_FLAT_DIM), dtype=np.float32)
    for ch_idx, ch in enumerate(["GoldsteinScale", "AvgTone", "NumMentions"]):
        flat_idx = ch_idx * N_NODES * N_NODES + fi * N_NODES + ti
        arr[row_idx, flat_idx] = raw[ch].values.astype(np.float32)

    channel_short = {"GoldsteinScale": "gs", "AvgTone": "at", "NumMentions": "nm"}
    col_names = [
        f"{channel_short[ch]}_{i}_{j}"
        for ch in ["GoldsteinScale", "AvgTone", "NumMentions"]
        for i in range(N_NODES)
        for j in range(N_NODES)
    ]
    flat_df = pd.DataFrame(arr, index=pd.DatetimeIndex(dates), columns=col_names)
    flat_df = flat_df.sort_index()

    if prices_df is not None:
        flat_df = flat_df.reindex(prices_df.index, method="ffill").fillna(0.0)

    return flat_df.astype(np.float32)


def _train_torch(model, X_tr, y_tr, X_val, y_val,
                 epochs=EPOCHS, patience=PATIENCE,
                 lr=LR, batch_size=BATCH_SIZE, name="model"):
    """Generic PyTorch training loop with early stopping."""
    model.to(DEVICE)
    opt  = torch.optim.Adam(model.parameters(), lr=lr)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()

    tr_dl  = DataLoader(_SimpleDataset(X_tr, y_tr),
                        batch_size=batch_size, shuffle=True,  num_workers=0)
    val_dl = DataLoader(_SimpleDataset(X_val, y_val),
                        batch_size=batch_size, shuffle=False, num_workers=0)

    best_loss    = float("inf")
    patience_cnt = 0
    best_state   = None

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_losses.append(crit(model(xb), yb).item())
        val_loss = float(np.mean(val_losses))
        sch.step(val_loss)

        if val_loss < best_loss:
            best_loss    = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  {name} early stop epoch {epoch} | best val {best_loss:.4f}")
                break

        if epoch % 20 == 0:
            print(f"  {name} epoch {epoch:3d} | val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class _SimpleDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


# =============================================================================
# GROUP 1 — Naive Baselines
# =============================================================================

# -----------------------------------------------------------------------------
# 1a. Random Walk
# -----------------------------------------------------------------------------

class RandomWalk:
    name = "RandomWalk"

    def fit(self, train_prices: np.ndarray, **_):
        pass

    def predict(self, test_prices: np.ndarray, lookback: int = 1, **_) -> np.ndarray:
        """y_hat_{t+1} = y_t (last observed price in each window)."""
        _, y = sliding_windows(test_prices, lookback)
        preds = test_prices[lookback - 1 : len(test_prices) - 1]
        return preds[: len(y)].astype(np.float32)


# -----------------------------------------------------------------------------
# 1b. Historical Mean
# -----------------------------------------------------------------------------

class HistoricalMeanBaseline:
    """Predict using the rolling mean of the last k days."""
    name = "HistoricalMean"

    def fit(self, train_prices: np.ndarray, **_):
        pass

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        """y_hat_{t+1} = mean(y_{t-k+1} ... y_t)."""
        X, _ = sliding_windows(test_prices, lookback)
        # X shape: [N, lookback, 5] -- mean over time axis
        preds = X.mean(axis=1)   # [N, 5]
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 1c. ARIMA
# -----------------------------------------------------------------------------

class ARIMABaseline:
    name = "ARIMA"

    def fit(self, train_prices: np.ndarray, **_):
        self._train = train_prices

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        _, y_true = sliding_windows(test_prices, lookback)
        n_test = len(y_true)
        preds  = np.zeros((n_test, N_NODES), dtype=np.float32)

        for node in range(N_NODES):
            full_series = np.concatenate([self._train[:, node],
                                          test_prices[:, node]])
            train_len   = len(self._train)
            node_preds  = []

            for i in range(lookback, lookback + n_test):
                history = full_series[:train_len + (i - lookback)]
                try:
                    model = ARIMA(history, order=(5, 1, 0))
                    fit   = model.fit()
                    fc    = fit.forecast(steps=1)
                    node_preds.append(float(fc[0]))
                except Exception:
                    node_preds.append(history[-1])

            preds[:, node] = node_preds
            print(f"  ARIMA node {node}/{N_NODES - 1} done")

        return preds


# -----------------------------------------------------------------------------
# 1d. ARIMAX (ARIMA with GDELT channels as exogenous covariates)
# -----------------------------------------------------------------------------

class ARIMAXBaseline:
    """
    ARIMAX: ARIMA(5,1,0) with GDELT AvgTone + GoldsteinScale per (from, to)
    pair as exogenous variables.  We use only the columns corresponding to the
    5 'row' interactions for each benchmark node (i.e. rows where to_node ==
    this benchmark) to keep the exog dimension manageable (10 features per
    node).
    """
    name = "ARIMAX"

    def fit(self, train_prices: np.ndarray, gdelt_flat_train: np.ndarray = None, **_):
        self._train      = train_prices
        self._gdelt_tr   = gdelt_flat_train  # [n_train, 75] or None

    def predict(self, test_prices: np.ndarray, lookback: int = 20,
                gdelt_flat_test: np.ndarray = None, **_) -> np.ndarray:
        from statsmodels.tsa.arima.model import ARIMA
        _, y_true = sliding_windows(test_prices, lookback)
        n_test    = len(y_true)
        preds     = np.zeros((n_test, N_NODES), dtype=np.float32)

        # If no GDELT provided, fall back to plain ARIMA
        if self._gdelt_tr is None or gdelt_flat_test is None:
            print("  ARIMAX: no GDELT data, falling back to plain ARIMA")
            plain = ARIMABaseline()
            plain.fit(self._train)
            return plain.predict(test_prices, lookback=lookback)

        for node in range(N_NODES):
            # Select exog cols: AvgTone (channel index 1) rows where to_node==node
            # col offset for AvgTone: 1 * 25 + from_node * 5 + to_node
            exog_cols_gs = [0 * N_NODES * N_NODES + fr * N_NODES + node
                            for fr in range(N_NODES)]   # GoldsteinScale -> node
            exog_cols_at = [1 * N_NODES * N_NODES + fr * N_NODES + node
                            for fr in range(N_NODES)]   # AvgTone -> node
            exog_cols    = exog_cols_gs + exog_cols_at  # 10 columns

            price_full   = np.concatenate([self._train[:, node],
                                           test_prices[:, node]])
            gdelt_full   = np.vstack([self._gdelt_tr, gdelt_flat_test])
            exog_full    = gdelt_full[:, exog_cols]
            
            # Shift exog by 1 step to use t-1 data for predicting t
            exog_shifted = np.zeros_like(exog_full)
            if len(exog_full) > 1:
                exog_shifted[1:] = exog_full[:-1]
                exog_shifted[0]  = exog_full[0]
            else:
                exog_shifted = exog_full.copy()
                
            train_len    = len(self._train)
            node_preds   = []

            for i in range(lookback, lookback + n_test):
                hist_price = price_full[:train_len + (i - lookback)]
                hist_exog  = exog_shifted[:train_len + (i - lookback)]
                next_exog  = exog_shifted[train_len + (i - lookback): train_len + (i - lookback) + 1]
                try:
                    model = ARIMA(hist_price, exog=hist_exog, order=(5, 1, 0))
                    fit   = model.fit()
                    fc    = fit.forecast(steps=1, exog=next_exog)
                    node_preds.append(float(fc[0]))
                except Exception:
                    node_preds.append(hist_price[-1])

            preds[:, node] = node_preds
            print(f"  ARIMAX node {node}/{N_NODES - 1} done")

        return preds


# =============================================================================
# GROUP 2 — ML Baselines Without Graph Structure
# =============================================================================

# -----------------------------------------------------------------------------
# 2a. XGBoost
# -----------------------------------------------------------------------------

class XGBoostBaseline:
    name = "XGBoost"

    def fit(self, train_prices: np.ndarray, lookback: int = 20, **_):
        from xgboost import XGBRegressor
        X, y   = sliding_windows(train_prices, lookback)
        X_flat = X.reshape(len(X), -1)
        self._models = []
        for node in range(N_NODES):
            reg = XGBRegressor(n_estimators=300, max_depth=6,
                               learning_rate=0.05, subsample=0.8,
                               random_state=42, n_jobs=1, verbosity=0)
            reg.fit(X_flat, y[:, node])
            self._models.append(reg)

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _   = sliding_windows(test_prices, lookback)
        X_flat = X.reshape(len(X), -1)
        preds  = np.column_stack([m.predict(X_flat) for m in self._models])
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 2b. SVR (Support Vector Regression)
# -----------------------------------------------------------------------------

class SVRBaseline:
    """Support Vector Regression per benchmark, standard kernel RBF."""
    name = "SVR"

    def fit(self, train_prices: np.ndarray, lookback: int = 20, **_):
        from sklearn.svm import SVR
        from sklearn.preprocessing import StandardScaler
        X, y         = sliding_windows(train_prices, lookback)
        X_flat       = X.reshape(len(X), -1)
        self._scalers_x = []
        self._scalers_y = []
        self._models    = []

        for node in range(N_NODES):
            sx = StandardScaler()
            sy = StandardScaler()
            Xs = sx.fit_transform(X_flat)
            ys = sy.fit_transform(y[:, node].reshape(-1, 1)).ravel()
            reg = SVR(kernel="rbf", C=10.0, epsilon=0.1, gamma="scale")
            reg.fit(Xs, ys)
            self._scalers_x.append(sx)
            self._scalers_y.append(sy)
            self._models.append(reg)

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _   = sliding_windows(test_prices, lookback)
        X_flat = X.reshape(len(X), -1)
        preds  = []
        for node in range(N_NODES):
            Xs   = self._scalers_x[node].transform(X_flat)
            yhat = self._models[node].predict(Xs)
            yhat = self._scalers_y[node].inverse_transform(yhat.reshape(-1, 1)).ravel()
            preds.append(yhat)
        return np.column_stack(preds).astype(np.float32)


# -----------------------------------------------------------------------------
# 2c. Vanilla LSTM (prices only, no GDELT, no graph)
# -----------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, input_size=N_NODES, hidden=128, num_layers=2,
                 dropout=DROPOUT, output_size=N_NODES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class LSTMBaseline:
    name = "LSTM"

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            lookback: int = 20, **_):
        X_tr,  y_tr  = sliding_windows(train_prices, lookback)
        X_val, y_val = sliding_windows(val_prices,   lookback)
        self._model  = _LSTMNet()
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="LSTM")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _ = sliding_windows(test_prices, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 2d. LSTM + GDELT flat (LSTM with GDELT features appended, no graph)
# -----------------------------------------------------------------------------

class _LSTMGDELTNet(nn.Module):
    """LSTM taking [prices (5) + gdelt_flat (75)] = 80-dim input per timestep."""
    def __init__(self, input_size=N_NODES + GDELT_FLAT_DIM,
                 hidden=128, num_layers=2,
                 dropout=DROPOUT, output_size=N_NODES):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, output_size)
        )

    def forward(self, x):
        # x: [B, lookback, 80]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


def _sliding_windows_with_gdelt(prices: np.ndarray, gdelt_flat: np.ndarray,
                                 lookback: int):
    """
    Build windows over concatenated [prices | gdelt] series.

    Parameters
    ----------
    prices     : [T, 5]
    gdelt_flat : [T, 75]  aligned to same dates as prices

    Returns
    -------
    X : [N, lookback, 80]
    y : [N, 5]
    """
    combined = np.concatenate([prices, gdelt_flat], axis=1).astype(np.float32)
    X, y = [], []
    for i in range(lookback, len(combined)):
        X.append(combined[i - lookback:i])
        y.append(prices[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


class LSTMGDELTBaseline:
    """
    LSTM with GDELT flat features (80-dim input: 5 prices + 75 GDELT).
    No graph structure — tests whether graph topology adds value over
    flat GDELT injection.
    """
    name = "LSTM_GDELT"

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            gdelt_flat_train: np.ndarray, gdelt_flat_val: np.ndarray,
            lookback: int = 20, **_):
        X_tr,  y_tr  = _sliding_windows_with_gdelt(train_prices, gdelt_flat_train, lookback)
        X_val, y_val = _sliding_windows_with_gdelt(val_prices,   gdelt_flat_val,   lookback)
        self._model  = _LSTMGDELTNet()
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="LSTM_GDELT")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray,
                gdelt_flat_test: np.ndarray,
                lookback: int = 20, **_) -> np.ndarray:
        X, _ = _sliding_windows_with_gdelt(test_prices, gdelt_flat_test, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# =============================================================================
# GROUP 3 — Graph Baselines Without Dynamic Edges
# =============================================================================

# -----------------------------------------------------------------------------
# 3a. GCN + static adjacency
# -----------------------------------------------------------------------------

class _GCNLayer(nn.Module):
    """Single GCN layer: H' = sigma(A_hat H W)."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, a_hat: torch.Tensor) -> torch.Tensor:
        # x    : [B, N_nodes, in_features]
        # a_hat: [B, N_nodes, N_nodes] normalized adjacency
        return F.relu(torch.bmm(a_hat, x) @ self.weight)


def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalisation: D^{-1/2} A D^{-1/2} + I."""
    adj = adj + torch.eye(adj.size(-1), device=adj.device).unsqueeze(0)
    deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    deg_inv_sqrt = deg ** -0.5
    return deg_inv_sqrt * adj * deg_inv_sqrt.transpose(-1, -2)


class _GCNStaticNet(nn.Module):
    def __init__(self, d_model=D_MODEL, n_gcn_layers=2,
                 dropout=DROPOUT, lookback=20):
        super().__init__()
        self.price_embed = nn.Linear(1, d_model)
        self.gcn_layers  = nn.ModuleList(
            [_GCNLayer(d_model, d_model) for _ in range(n_gcn_layers)]
        )
        self.dropout     = nn.Dropout(dropout)
        # After GCN per timestep, pool over nodes then feed GRU
        self.gru         = nn.GRU(N_NODES * d_model, d_model,
                                  num_layers=2, batch_first=True,
                                  dropout=dropout)
        self.head        = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, N_NODES)
        )

    def forward(self, prices: torch.Tensor, a_static: torch.Tensor):
        # prices  : [B, lookback, 5]
        # a_static: [B, 5, 5]
        B, k, _ = prices.shape
        a_hat   = _normalize_adj(a_static)

        seq = []
        for t in range(k):
            # [B, 5, 1] -> embed -> [B, 5, d]
            h = self.price_embed(prices[:, t, :].unsqueeze(-1))
            for gcn in self.gcn_layers:
                h = self.dropout(gcn(h, a_hat))
            seq.append(h.reshape(B, -1))   # [B, 5*d]

        seq_t = torch.stack(seq, dim=1)    # [B, k, 5*d]
        _, h_n = self.gru(seq_t)           # h_n: [2, B, d]
        z = h_n[-1]                        # [B, d]
        return self.head(z)


class _StaticGraphDataset(Dataset):
    """Dataset that pairs price windows with a static adjacency matrix."""
    def __init__(self, prices_df, adj_df, lookback, start, end):
        prices       = prices_df.loc[start:end].values.astype(np.float32)
        self.X, self.y = sliding_windows(prices, lookback)
        dates        = prices_df.loc[start:end].index[lookback-1:-1]
        adj_cols     = [f"col_{r}_{c}" for r in range(N_NODES)
                        for c in range(N_NODES)]
        adjs = []
        for d in dates:
            period = d.year * 100 + d.month
            row    = adj_df[adj_df["period"] == period]
            if row.empty:
                row = adj_df[adj_df["period"] <= period].iloc[-1:]
            if row.empty:
                adjs.append(np.eye(N_NODES, dtype=np.float32) / N_NODES)
            else:
                adjs.append(row[adj_cols].values.reshape(N_NODES, N_NODES))
        self.adjs = np.array(adjs, dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return (torch.tensor(self.X[i]),
                torch.tensor(self.adjs[i]),
                torch.tensor(self.y[i]))


def _train_static_graph(model, tr_ds, val_ds, name="StaticGraph"):
    opt  = torch.optim.Adam(model.parameters(), lr=LR)
    sch  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    crit = nn.MSELoss()
    tr_dl  = DataLoader(tr_ds,  batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_loss, patience_cnt, best_state = float("inf"), 0, None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for X, adj, y in tr_dl:
            X, adj, y = X.to(DEVICE), adj.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(X, adj), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        vl = []
        with torch.no_grad():
            for X, adj, y in val_dl:
                X, adj, y = X.to(DEVICE), adj.to(DEVICE), y.to(DEVICE)
                vl.append(crit(model(X, adj), y).item())
        val_loss = float(np.mean(vl))
        sch.step(val_loss)

        if val_loss < best_loss:
            best_loss    = val_loss
            best_state   = {k: v.clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"  {name} early stop epoch {epoch} | best val {best_loss:.4f}")
                break

        if epoch % 20 == 0:
            print(f"  {name} epoch {epoch:3d} | val {val_loss:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


class GCNStaticBaseline:
    """Standard GCN on fixed Comtrade matrix, no GDELT."""
    name = "GCNStatic"

    def fit(self, prices_df: pd.DataFrame, adj_df: pd.DataFrame,
            lookback: int = 20, **_):
        tr_ds  = _StaticGraphDataset(prices_df, adj_df, lookback,
                                     TRAIN_START, TRAIN_END)
        val_ds = _StaticGraphDataset(prices_df, adj_df, lookback,
                                     VAL_START, VAL_END)
        model  = _GCNStaticNet(lookback=lookback).to(DEVICE)
        self._model    = _train_static_graph(model, tr_ds, val_ds, name="GCNStatic")
        self._lookback = lookback

    def predict(self, prices_df: pd.DataFrame, adj_df: pd.DataFrame,
                lookback: int = 20, **_) -> np.ndarray:
        te_ds = _StaticGraphDataset(prices_df, adj_df, lookback,
                                    TEST_START, "2026-12-31")
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self._model.eval()
        preds = []
        with torch.no_grad():
            for X, adj, _ in te_dl:
                X, adj = X.to(DEVICE), adj.to(DEVICE)
                preds.append(self._model(X, adj).cpu().numpy())
        return np.vstack(preds).astype(np.float32)


# -----------------------------------------------------------------------------
# 3b. GAT + static adjacency (renamed from StaticGATBaseline)
# -----------------------------------------------------------------------------

class _GATLayer(nn.Module):
    """Multi-head Graph Attention Layer."""
    def __init__(self, in_features: int, out_features: int, n_heads: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        assert out_features % n_heads == 0
        self.n_heads    = n_heads
        self.d_head     = out_features // n_heads
        self.W          = nn.Linear(in_features, out_features, bias=False)
        self.a          = nn.Parameter(torch.empty(n_heads, 2 * self.d_head))
        nn.init.xavier_uniform_(self.a.unsqueeze(0))
        self.dropout    = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x  : [B, N, in_features]
        # adj: [B, N, N]
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.n_heads, self.d_head)  # [B,N,H,D]
        h = h.permute(0, 2, 1, 3)                            # [B,H,N,D]

        e_src = (h * self.a[:, :self.d_head]).sum(-1, keepdim=True)   # [B,H,N,1]
        e_dst = (h * self.a[:, self.d_head:]).sum(-1, keepdim=True)   # [B,H,N,1]
        e     = self.leaky_relu(e_src + e_dst.transpose(-2, -1))       # [B,H,N,N]

        mask  = (adj.unsqueeze(1) == 0)
        e     = e.masked_fill(mask, float("-inf"))
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)                                    # [B,H,N,N]

        out = torch.matmul(alpha, h).permute(0, 2, 1, 3)              # [B,N,H,D]
        return F.elu(out.reshape(B, N, -1))                            # [B,N,out_features]


class _GATStaticNet(nn.Module):
    """GAT + GRU using fixed A_static only — no GDELT edge gating."""
    def __init__(self, d_model=D_MODEL, n_heads_gat=N_HEADS_GAT,
                 dropout=DROPOUT, lookback=20):
        super().__init__()
        self.price_embed = nn.Linear(1, d_model)
        self.gat         = _GATLayer(d_model, d_model, n_heads=n_heads_gat,
                                     dropout=dropout)
        self.gru         = nn.GRU(N_NODES * d_model, d_model,
                                  num_layers=2, batch_first=True,
                                  dropout=dropout)
        self.head        = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, N_NODES)
        )

    def forward(self, prices: torch.Tensor, a_static: torch.Tensor):
        B, k, _ = prices.shape
        seq = []
        for t in range(k):
            h_t = self.price_embed(prices[:, t, :].unsqueeze(-1))   # [B, 5, d]
            h_t = self.gat(h_t, a_static)                           # [B, 5, d]
            seq.append(h_t.reshape(B, -1))
        seq_t = torch.stack(seq, dim=1)                              # [B, k, 5*d]
        _, h_n = self.gru(seq_t)
        return self.head(h_n[-1])


class GATStaticBaseline:
    """GAT with fixed Comtrade A_static, no GDELT gating."""
    name = "GATStatic"

    def fit(self, prices_df: pd.DataFrame, adj_df: pd.DataFrame,
            lookback: int = 20, **_):
        tr_ds  = _StaticGraphDataset(prices_df, adj_df, lookback,
                                     TRAIN_START, TRAIN_END)
        val_ds = _StaticGraphDataset(prices_df, adj_df, lookback,
                                     VAL_START, VAL_END)
        model  = _GATStaticNet(lookback=lookback).to(DEVICE)
        self._model    = _train_static_graph(model, tr_ds, val_ds, name="GATStatic")
        self._lookback = lookback

    def predict(self, prices_df: pd.DataFrame, adj_df: pd.DataFrame,
                lookback: int = 20, **_) -> np.ndarray:
        te_ds = _StaticGraphDataset(prices_df, adj_df, lookback,
                                    TEST_START, "2026-12-31")
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self._model.eval()
        preds = []
        with torch.no_grad():
            for X, adj, _ in te_dl:
                X, adj = X.to(DEVICE), adj.to(DEVICE)
                preds.append(self._model(X, adj).cpu().numpy())
        return np.vstack(preds).astype(np.float32)


# Backward compatibility alias
StaticGATBaseline = GATStaticBaseline


# -----------------------------------------------------------------------------
# 3c. DCRNN — Diffusion Convolutional Recurrent Neural Network
# -----------------------------------------------------------------------------

class _DiffConv(nn.Module):
    """
    Bidirectional diffusion graph convolution.
    Approximates graph convolution via K-step random walks on
    the forward (A) and backward (A^T) transition matrices.
    """
    def __init__(self, in_features: int, out_features: int,
                 K: int = 3, bias: bool = True):
        super().__init__()
        self.K = K
        # Weight matrix for forward + backward + self: (2K+1) terms
        self.weight = nn.Parameter(
            torch.empty(2 * K + 1, in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None
        nn.init.xavier_uniform_(self.weight.view(-1, out_features))

    @staticmethod
    def _transition(adj: torch.Tensor) -> torch.Tensor:
        """Row-normalise adjacency to get transition matrix."""
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return adj / deg

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x  : [B, N, in_features]
        # adj: [B, N, N]
        T_fwd = self._transition(adj)          # D_out^{-1} A
        T_bwd = self._transition(adj.transpose(-1, -2))  # D_in^{-1} A^T

        supports = []
        x_pow_f = x
        x_pow_b = x
        supports.append(x)   # K=0 term (identity)

        for _ in range(self.K):
            x_pow_f = torch.bmm(T_fwd, x_pow_f)
            x_pow_b = torch.bmm(T_bwd, x_pow_b)
            supports.append(x_pow_f)
            supports.append(x_pow_b)

        # supports: list of (2K+1) tensors each [B, N, in_features]
        h = torch.stack(supports, dim=0)       # [2K+1, B, N, in_f]
        h = h.permute(1, 2, 0, 3)             # [B, N, 2K+1, in_f]

        # Contract: sum over K terms and in_features
        # weight: [2K+1, in_f, out_f]
        out = torch.einsum("bnki,kio->bno", h, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out


class _DCGRUCell(nn.Module):
    """Single DCGRU cell using DiffConv in place of linear transforms."""
    def __init__(self, d_model: int, K: int = 3):
        super().__init__()
        self.d_model = d_model
        # Gates: reset, update (uses [x || h])
        self.conv_r = _DiffConv(d_model * 2, d_model, K=K)
        self.conv_u = _DiffConv(d_model * 2, d_model, K=K)
        self.conv_c = _DiffConv(d_model * 2, d_model, K=K)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                adj: torch.Tensor) -> torch.Tensor:
        # x, h : [B, N, d_model]
        # adj  : [B, N, N]
        xh = torch.cat([x, h], dim=-1)               # [B, N, 2d]
        r  = torch.sigmoid(self.conv_r(xh, adj))
        u  = torch.sigmoid(self.conv_u(xh, adj))
        xh_r = torch.cat([x, r * h], dim=-1)
        c  = torch.tanh(self.conv_c(xh_r, adj))
        return u * h + (1 - u) * c


class _DCRNNNet(nn.Module):
    """
    DCRNN encoder: stack of DCGRU cells unrolled over the lookback window,
    then a linear head per node that produces N_NODES forecasts.
    """
    def __init__(self, d_model: int = D_MODEL, n_layers: int = 2,
                 K: int = 3, dropout: float = DROPOUT):
        super().__init__()
        self.price_embed = nn.Linear(1, d_model)
        self.cells       = nn.ModuleList(
            [_DCGRUCell(d_model, K=K) for _ in range(n_layers)]
        )
        self.dropout     = nn.Dropout(dropout)
        # Output: pool over nodes (mean) then project
        self.head        = nn.Sequential(
            nn.Linear(N_NODES * d_model, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, N_NODES)
        )

    def forward(self, prices: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # prices: [B, lookback, 5]
        # adj   : [B, 5, 5]
        B, k, _ = prices.shape
        d = self.cells[0].d_model

        # Initialise hidden states
        hs = [torch.zeros(B, N_NODES, d, device=prices.device)
              for _ in self.cells]

        for t in range(k):
            x_t = self.price_embed(prices[:, t, :].unsqueeze(-1))  # [B,5,d]
            for layer_idx, cell in enumerate(self.cells):
                x_t = cell(x_t, hs[layer_idx], adj)
                x_t = self.dropout(x_t)
                hs[layer_idx] = x_t

        # Final hidden state of last layer: [B, 5, d] -> flatten -> head
        z = hs[-1].reshape(B, -1)     # [B, 5*d]
        return self.head(z)


class DCRNNBaseline:
    """Diffusion Convolutional Recurrent Neural Network baseline."""
    name = "DCRNN"

    def fit(self, prices_df: pd.DataFrame, adj_df: pd.DataFrame,
            lookback: int = 20, **_):
        tr_ds  = _StaticGraphDataset(prices_df, adj_df, lookback,
                                     TRAIN_START, TRAIN_END)
        val_ds = _StaticGraphDataset(prices_df, adj_df, lookback,
                                     VAL_START, VAL_END)
        model  = _DCRNNNet().to(DEVICE)
        self._model    = _train_static_graph(model, tr_ds, val_ds, name="DCRNN")
        self._lookback = lookback

    def predict(self, prices_df: pd.DataFrame, adj_df: pd.DataFrame,
                lookback: int = 20, **_) -> np.ndarray:
        te_ds = _StaticGraphDataset(prices_df, adj_df, lookback,
                                    TEST_START, "2026-12-31")
        te_dl = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self._model.eval()
        preds = []
        with torch.no_grad():
            for X, adj, _ in te_dl:
                X, adj = X.to(DEVICE), adj.to(DEVICE)
                preds.append(self._model(X, adj).cpu().numpy())
        return np.vstack(preds).astype(np.float32)


# =============================================================================
# GROUP 4 — Strong Temporal Baselines
# =============================================================================

# -----------------------------------------------------------------------------
# 4a. TFT — Temporal Fusion Transformer
# -----------------------------------------------------------------------------

class _GatedResidualNetwork(nn.Module):
    """GRN building block from the TFT paper."""
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.fc1   = nn.Linear(d_model, d_model)
        self.fc2   = nn.Linear(d_model, d_model)
        self.gate  = nn.Linear(d_model, d_model)
        self.ln    = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        g = torch.sigmoid(self.gate(x))
        return self.ln(x + g * h)


class _VariableSelectionNetwork(nn.Module):
    """Soft feature selection via GRN + softmax weights."""
    def __init__(self, n_features: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.input_proj  = nn.Linear(n_features, d_model)
        self.grns        = nn.ModuleList(
            [_GatedResidualNetwork(d_model, dropout) for _ in range(n_features)]
        )
        self.weight_grn  = _GatedResidualNetwork(d_model, dropout)
        self.weight_head = nn.Linear(d_model, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, n_features]  or  [B, n_features]
        squeezed = (x.dim() == 2)
        if squeezed:
            x = x.unsqueeze(1)   # [B, 1, n_features]

        B, T, F = x.shape
        # Per-feature transformation
        feat_list = []
        for i in range(min(F, len(self.grns))):
            xi = self.input_proj(x[:, :, i:i+1].expand(B, T, F))
            feat_list.append(self.grns[i](xi))
        # Stack: [B, T, F, d]  — here we just sum for simplicity
        stacked = torch.stack(feat_list, dim=2)  # [B, T, F, d]

        # Selection weights
        ctx      = self.input_proj(x)                     # [B, T, d]
        w_logits = self.weight_head(self.weight_grn(ctx)) # [B, T, F]
        w        = torch.softmax(w_logits, dim=-1).unsqueeze(-1)  # [B,T,F,1]
        out      = (stacked * w).sum(dim=2)               # [B, T, d]

        return out.squeeze(1) if squeezed else out


class _TFTNet(nn.Module):
    """
    Simplified Temporal Fusion Transformer.
    Components: variable selection, LSTM encoder/decoder, multi-head attention,
    gated residual post-attention, linear head.
    """
    def __init__(self, n_features: int = N_NODES,
                 d_model: int = D_MODEL, n_heads: int = 4,
                 n_lstm_layers: int = 2, dropout: float = DROPOUT,
                 lookback: int = 20):
        super().__init__()
        self.vsn         = _VariableSelectionNetwork(n_features, d_model, dropout)
        self.encoder     = nn.LSTM(d_model, d_model, n_lstm_layers,
                                   batch_first=True, dropout=dropout)
        self.decoder     = nn.LSTM(d_model, d_model, n_lstm_layers,
                                   batch_first=True, dropout=dropout)
        self.attn        = nn.MultiheadAttention(d_model, n_heads,
                                                  dropout=dropout,
                                                  batch_first=True)
        self.grn_attn    = _GatedResidualNetwork(d_model, dropout)
        self.gate_enc    = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.ln1         = nn.LayerNorm(d_model)
        self.head        = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, N_NODES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, lookback, n_features]
        enc_in         = self.vsn(x)                            # [B, k, d]
        enc_out, h_c   = self.encoder(enc_in)                   # [B, k, d]

        # One-step decoder query: last encoder hidden
        dec_in         = enc_out[:, -1:, :]                     # [B, 1, d]
        dec_out, _     = self.decoder(dec_in, h_c)              # [B, 1, d]

        # Multi-head self-attention over encoder output + decoder query
        query          = dec_out                                 # [B, 1, d]
        attn_out, _    = self.attn(query, enc_out, enc_out)     # [B, 1, d]
        attn_out       = self.grn_attn(attn_out.squeeze(1))     # [B, d]

        # Gated skip from decoder
        g              = self.gate_enc(dec_out.squeeze(1))
        z              = self.ln1(attn_out + g * dec_out.squeeze(1))
        return self.head(z)


class TFTBaseline:
    """Temporal Fusion Transformer baseline (simplified clean implementation)."""
    name = "TFT"

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            lookback: int = 20, **_):
        X_tr,  y_tr  = sliding_windows(train_prices, lookback)
        X_val, y_val = sliding_windows(val_prices,   lookback)
        self._model  = _TFTNet(lookback=lookback)
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="TFT")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _ = sliding_windows(test_prices, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 4b. PatchTST — Patch-based time-series Transformer
# -----------------------------------------------------------------------------

class _PatchTSTNet(nn.Module):
    """
    PatchTST: divide lookback window into non-overlapping patches,
    project each patch to d_model, apply Transformer encoder,
    then flatten + linear head.
    Reference: Nie et al. 2023.
    """
    def __init__(self, n_features: int = N_NODES,
                 d_model: int = D_MODEL, n_heads: int = 4,
                 n_layers: int = 3, dropout: float = DROPOUT,
                 lookback: int = 20, patch_len: int = 4):
        super().__init__()
        self.patch_len = patch_len
        self.n_patches = lookback // patch_len   # number of patches per channel
        remainder      = lookback % patch_len
        if remainder != 0:
            # Pad to make lookback divisible by patch_len
            self.pad = remainder
        else:
            self.pad = 0
        total_patches = (lookback + self.pad) // patch_len

        # Channel-independent: each of the n_features channels is patched separately
        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos_enc     = nn.Parameter(torch.randn(1, total_patches, d_model))
        enc_layer        = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder     = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.ln          = nn.LayerNorm(d_model)
        # Head: flatten all channels' patch representations
        self.head        = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_features * total_patches * d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, N_NODES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, lookback, n_features]
        B, k, C = x.shape
        if self.pad > 0:
            x = F.pad(x, (0, 0, 0, self.pad))  # pad time dim

        # Reshape into patches: [B, C, n_patches, patch_len]
        x = x.permute(0, 2, 1)                   # [B, C, k+pad]
        x = x.unfold(dimension=2, size=self.patch_len, step=self.patch_len)
        # x: [B, C, n_patches, patch_len]
        B, C, P, L = x.shape

        # Process each channel independently, reuse same Transformer
        x = x.reshape(B * C, P, L)               # [B*C, P, patch_len]
        h = self.patch_embed(x)                   # [B*C, P, d]
        h = h + self.pos_enc[:, :P, :]
        h = self.encoder(h)                       # [B*C, P, d]
        h = self.ln(h)                            # [B*C, P, d]
        h = h.reshape(B, C, P, -1)               # [B, C, P, d]
        return self.head(h)


class PatchTSTBaseline:
    """PatchTST Transformer baseline."""
    name = "PatchTST"

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            lookback: int = 20, **_):
        X_tr,  y_tr  = sliding_windows(train_prices, lookback)
        X_val, y_val = sliding_windows(val_prices,   lookback)
        self._model  = _PatchTSTNet(lookback=lookback)
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="PatchTST")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _ = sliding_windows(test_prices, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 4c. N-BEATS — Neural Basis Expansion Analysis for Interpretable TS Forecasting
# -----------------------------------------------------------------------------

class _NBEATSBlock(nn.Module):
    """
    Single N-BEATS block.
    Produces backcast (reconstruction of input) and forecast (future prediction).
    """
    def __init__(self, lookback: int, forecast_len: int,
                 hidden: int = 256, n_layers: int = 4,
                 basis_type: str = "generic"):
        super().__init__()
        self.lookback     = lookback
        self.forecast_len = forecast_len
        self.basis_type   = basis_type

        layers = []
        in_size = lookback
        for _ in range(n_layers):
            layers += [nn.Linear(in_size, hidden), nn.ReLU()]
            in_size = hidden
        self.fc_stack = nn.Sequential(*layers)

        # Generic basis: direct linear projections
        self.backcast_head  = nn.Linear(hidden, lookback)
        self.forecast_head  = nn.Linear(hidden, forecast_len)

    def forward(self, x: torch.Tensor):
        # x: [B, lookback]  (single channel)
        h         = self.fc_stack(x)
        backcast  = self.backcast_head(h)
        forecast  = self.forecast_head(h)
        return backcast, forecast


class _NBEATSStack(nn.Module):
    """Stack of N-BEATS blocks with residual connections."""
    def __init__(self, lookback: int, forecast_len: int,
                 n_blocks: int = 3, hidden: int = 256,
                 n_layers: int = 4, basis_type: str = "generic"):
        super().__init__()
        self.blocks = nn.ModuleList([
            _NBEATSBlock(lookback, forecast_len, hidden, n_layers, basis_type)
            for _ in range(n_blocks)
        ])

    def forward(self, x: torch.Tensor):
        # x: [B, lookback]
        residual     = x
        stack_fcst   = torch.zeros(x.size(0), self.blocks[0].forecast_len,
                                   device=x.device)
        for block in self.blocks:
            backcast, forecast = block(residual)
            residual   = residual - backcast
            stack_fcst = stack_fcst + forecast
        return stack_fcst


class _NBEATSNet(nn.Module):
    """
    Full N-BEATS model with trend + seasonality + generic stacks per channel.
    Operates channel-independently then aggregates.
    """
    def __init__(self, lookback: int = 20, n_channels: int = N_NODES,
                 hidden: int = 256, n_blocks_per_stack: int = 3,
                 dropout: float = DROPOUT):
        super().__init__()
        self.n_channels  = n_channels
        forecast_len     = 1

        # One generic stack per channel (can be extended to trend+seasonality)
        self.stacks = nn.ModuleList([
            _NBEATSStack(lookback, forecast_len,
                         n_blocks=n_blocks_per_stack,
                         hidden=hidden)
            for _ in range(n_channels)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, lookback, n_channels]
        outs = []
        for c in range(self.n_channels):
            xc    = x[:, :, c]             # [B, lookback]
            fcst  = self.stacks[c](xc)     # [B, 1]
            outs.append(fcst)
        return torch.cat(outs, dim=-1)     # [B, n_channels]


class NBEATSBaseline:
    """N-BEATS Neural Basis Expansion baseline."""
    name = "N-BEATS"

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            lookback: int = 20, **_):
        X_tr,  y_tr  = sliding_windows(train_prices, lookback)
        X_val, y_val = sliding_windows(val_prices,   lookback)
        self._model  = _NBEATSNet(lookback=lookback)
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="N-BEATS")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _ = sliding_windows(test_prices, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# =============================================================================
# GROUP 5 — Domain-Specific Prior Work
# =============================================================================

# -----------------------------------------------------------------------------
# 5a. EMD-LSTM — Empirical Mode Decomposition + LSTM
# -----------------------------------------------------------------------------

class _EMDLSTMNet(nn.Module):
    """LSTM that accepts n_imf IMFs per channel as input."""
    def __init__(self, input_size: int, hidden: int = 128,
                 num_layers: int = 2, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, N_NODES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, lookback, input_size]
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])


class EMDLSTMBaseline:
    """
    EMD-LSTM: Empirical Mode Decomposition + LSTM.
    Each price channel is decomposed into IMFs using PyEMD.
    The IMFs are stacked as additional input features for the LSTM.

    Requires: pip install EMD-signal
    If PyEMD is unavailable, falls back to vanilla LSTM.
    """
    name = "EMD-LSTM"

    def __init__(self, n_imf: int = 4):
        self.n_imf  = n_imf
        self._model = None

    def _decompose(self, prices: np.ndarray) -> np.ndarray:
        """
        Decompose each channel into n_imf IMFs.

        Parameters
        ----------
        prices : [T, 5]

        Returns
        -------
        features : [T, 5 * (n_imf + 1)]
          original 5 channels + n_imf IMFs per channel
        """
        try:
            from PyEMD import EMD
        except ImportError:
            print("  EMD-LSTM: PyEMD not installed. Returning prices as-is.")
            return prices

        T, C = prices.shape
        decomposed = [prices]   # keep original as first feature block

        for c in range(C):
            emd  = EMD()
            imfs = emd.emd(prices[:, c], max_imf=self.n_imf)
            # Pad or truncate to exactly n_imf components
            if imfs.shape[0] < self.n_imf:
                pad   = np.zeros((self.n_imf - imfs.shape[0], T), dtype=np.float32)
                imfs  = np.vstack([imfs, pad])
            else:
                imfs  = imfs[:self.n_imf]
            decomposed.append(imfs.T)   # [T, n_imf]

        return np.hstack(decomposed).astype(np.float32)  # [T, 5*(n_imf+1)]

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            lookback: int = 20, **_):
        tr_feat  = self._decompose(train_prices)
        val_feat = self._decompose(val_prices)
        input_sz = tr_feat.shape[1]

        X_tr,  y_tr  = sliding_windows(tr_feat,  lookback)
        X_val, y_val = sliding_windows(val_feat, lookback)
        # y must be prices-only; use original prices as target
        _, y_tr_p  = sliding_windows(train_prices, lookback)
        _, y_val_p = sliding_windows(val_prices,   lookback)

        self._model  = _EMDLSTMNet(input_size=input_sz)
        self._model  = _train_torch(self._model,
                                    X_tr, y_tr_p, X_val, y_val_p,
                                    name="EMD-LSTM")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        te_feat = self._decompose(test_prices)
        X, _    = sliding_windows(te_feat, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 5b. Sentiment-LSTM — LSTM with GDELT AvgTone as flat sentiment feature
# -----------------------------------------------------------------------------

class SentimentLSTMBaseline:
    """
    LSTM with GDELT AvgTone as flat sentiment feature.
    Input: [prices (5) + AvgTone_flat (25)] = 30 features per timestep.

    This tests whether graph topology adds value over raw sentiment injection.
    Note: For the full Caldara-Iacoviello GPR index version, replace
    AvgTone with the official GPR index series (see GPRLSTMBaseline below).
    """
    name = "SentimentLSTM"
    _AVGTONE_DIM = N_NODES * N_NODES   # 25 (5x5 AvgTone matrix, flattened)

    def _extract_avgtone(self, gdelt_flat: np.ndarray) -> np.ndarray:
        """Extract AvgTone columns (channel index 1: cols 25..49 in 75-col vector)."""
        start = 1 * N_NODES * N_NODES   # = 25
        end   = 2 * N_NODES * N_NODES   # = 50
        return gdelt_flat[:, start:end].astype(np.float32)

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            gdelt_flat_train: np.ndarray, gdelt_flat_val: np.ndarray,
            lookback: int = 20, **_):
        at_tr  = self._extract_avgtone(gdelt_flat_train)   # [T, 25]
        at_val = self._extract_avgtone(gdelt_flat_val)

        feat_tr  = np.hstack([train_prices, at_tr]).astype(np.float32)
        feat_val = np.hstack([val_prices,   at_val]).astype(np.float32)

        X_tr,  _ = sliding_windows(feat_tr,  lookback)
        X_val, _ = sliding_windows(feat_val, lookback)
        _, y_tr  = sliding_windows(train_prices, lookback)
        _, y_val = sliding_windows(val_prices,   lookback)

        input_sz     = feat_tr.shape[1]
        self._model  = _EMDLSTMNet(input_size=input_sz)
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="SentimentLSTM")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, gdelt_flat_test: np.ndarray,
                lookback: int = 20, **_) -> np.ndarray:
        at_te   = self._extract_avgtone(gdelt_flat_test)
        feat_te = np.hstack([test_prices, at_te]).astype(np.float32)
        X, _    = sliding_windows(feat_te, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# -----------------------------------------------------------------------------
# 5c. GPR-LSTM — Geopolitical Risk Index proxy + LSTM
# -----------------------------------------------------------------------------

class GPRLSTMBaseline:
    """
    Geopolitical Risk (GPR) + LSTM baseline.

    Implements the Caldara & Iacoviello (2022) GPR-model approach.
    Since the official GPR index data requires a separate download
    (https://www.matteoiacoviello.com/gpr.htm), this implementation uses
    GDELT AvgTone as a proxy for geopolitical risk sentiment when the
    official GPR series is not available.

    To use the official GPR index:
        gpr_baseline.load_gpr_index(path_to_gpr_csv)
    where the CSV has columns [date, GPR_Global].

    Input: [prices (5) + GPR_proxy (1)] = 6 features per timestep
    (or more if the official multi-series GPR is used).

    Reference:
        Caldara, D. and Iacoviello, M. (2022). Measuring Geopolitical Risk.
        American Economic Review, 112(4), 1194-1225.
    """
    name = "GPR-LSTM"

    def __init__(self):
        self._gpr_series = None

    def load_gpr_index(self, gpr_csv_path: str):
        """
        Load official Caldara-Iacoviello GPR index from CSV.
        Expected columns: [date, GPR_Global] (or 'gpr', 'GPR', etc.)
        """
        df = pd.read_csv(gpr_csv_path, parse_dates=["date"])
        df = df.set_index("date").sort_index()
        # Normalise column name
        gpr_col = [c for c in df.columns
                   if c.lower().startswith("gpr")]
        if gpr_col:
            self._gpr_series = df[gpr_col[0]].rename("GPR")
        else:
            raise ValueError("No GPR column found in CSV: " + str(df.columns.tolist()))
        print(f"  GPR-LSTM: loaded official GPR index from {gpr_csv_path}")

    def _get_gpr_feature(self, prices_dates: pd.DatetimeIndex,
                         gdelt_flat: np.ndarray) -> np.ndarray:
        """Return [T, 1] GPR feature — official if available, else GDELT proxy."""
        if self._gpr_series is not None:
            aligned = self._gpr_series.reindex(prices_dates, method="ffill").fillna(0.0)
            vals    = aligned.values.reshape(-1, 1).astype(np.float32)
            # Normalise to ~[-1, 1]
            mu, sd  = vals.mean(), vals.std() + 1e-8
            return (vals - mu) / sd
        else:
            # Proxy: mean AvgTone across all node pairs (col 25:50)
            avgtone = gdelt_flat[:, 25:50]
            proxy   = avgtone.mean(axis=1, keepdims=True).astype(np.float32)
            mu, sd  = proxy.mean(), proxy.std() + 1e-8
            return (proxy - mu) / sd

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            gdelt_flat_train: np.ndarray, gdelt_flat_val: np.ndarray,
            train_dates: pd.DatetimeIndex = None,
            val_dates: pd.DatetimeIndex = None,
            lookback: int = 20, **_):

        if self._gpr_series is None:
            print("  GPR-LSTM: official GPR index not loaded; "
                  "using GDELT AvgTone proxy as geopolitical risk signal.")

        gpr_tr  = self._get_gpr_feature(
            train_dates if train_dates is not None
            else pd.date_range("2010-01-01", periods=len(train_prices), freq="B"),
            gdelt_flat_train)
        gpr_val = self._get_gpr_feature(
            val_dates if val_dates is not None
            else pd.date_range("2020-01-01", periods=len(val_prices), freq="B"),
            gdelt_flat_val)

        feat_tr  = np.hstack([train_prices, gpr_tr]).astype(np.float32)
        feat_val = np.hstack([val_prices,   gpr_val]).astype(np.float32)

        X_tr,  _ = sliding_windows(feat_tr,  lookback)
        X_val, _ = sliding_windows(feat_val, lookback)
        _, y_tr  = sliding_windows(train_prices, lookback)
        _, y_val = sliding_windows(val_prices,   lookback)

        input_sz     = feat_tr.shape[1]
        self._model  = _EMDLSTMNet(input_size=input_sz)
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="GPR-LSTM")
        self._lookback = lookback
        self._input_sz = input_sz

    def predict(self, test_prices: np.ndarray,
                gdelt_flat_test: np.ndarray,
                test_dates: pd.DatetimeIndex = None,
                lookback: int = 20, **_) -> np.ndarray:
        gpr_te  = self._get_gpr_feature(
            test_dates if test_dates is not None
            else pd.date_range("2022-01-01", periods=len(test_prices), freq="B"),
            gdelt_flat_test)
        feat_te = np.hstack([test_prices, gpr_te]).astype(np.float32)
        X, _    = sliding_windows(feat_te, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# =============================================================================
# LEGACY: Vanilla Transformer (kept from original baselines.py)
# =============================================================================

class _VanillaTransformer(nn.Module):
    def __init__(self, input_size=N_NODES, d_model=D_MODEL,
                 n_heads=4, n_layers=2, dropout=DROPOUT):
        super().__init__()
        self.proj    = nn.Linear(input_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, 200, d_model))
        enc_layer    = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head    = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64), nn.ReLU(),
            nn.Linear(64, N_NODES)
        )

    def forward(self, x):
        k = x.size(1)
        h = self.proj(x) + self.pos_enc[:, :k, :]
        z = self.encoder(h)[:, -1, :]
        return self.head(z)


class TransformerBaseline:
    name = "Transformer"

    def fit(self, train_prices: np.ndarray, val_prices: np.ndarray,
            lookback: int = 20, **_):
        X_tr, y_tr   = sliding_windows(train_prices, lookback)
        X_val, y_val = sliding_windows(val_prices,   lookback)
        self._model  = _VanillaTransformer()
        self._model  = _train_torch(self._model, X_tr, y_tr, X_val, y_val,
                                    name="Transformer")
        self._lookback = lookback

    def predict(self, test_prices: np.ndarray, lookback: int = 20, **_) -> np.ndarray:
        X, _ = sliding_windows(test_prices, lookback)
        self._model.eval()
        with torch.no_grad():
            preds = self._model(torch.tensor(X).to(DEVICE)).cpu().numpy()
        return preds.astype(np.float32)


# =============================================================================
# Runner
# =============================================================================


def evaluate_and_save(model_name, y_true, y_pred, test_dates_window, lookback):
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
    
    out_dir = RESULTS_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(out_dir / f"metrics_k{lookback}.json", "w") as f:
        json.dump(full_res, f, indent=2)
        
    df_pred = pd.DataFrame(y_pred, index=test_dates_window, columns=NODES)
    df_pred.to_csv(out_dir / f"predictions_k{lookback}.csv")
    
    return res_overall

def run_all_baselines(lookback: int = LOOKBACK_WINDOW, skip_arima: bool = False):
    print("=" * 60)
    print(f"Running all baselines  |  lookback={lookback}  |  device={DEVICE}")
    print("=" * 60)

    # Load from pre-split directories
    def _load_split(split_dir):
        df = pd.read_parquet(Path(split_dir) / "prices.parquet")
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        return df

    train_df = _load_split(TRAIN_DIR)
    val_df   = _load_split(VAL_DIR)
    test_df  = _load_split(TEST_DIR)
    prices_df = pd.concat([train_df, val_df, test_df]).sort_index()

    # Load adjacency — try split dirs or global
    adj_path = DATA_DIR / "uncomtrade" / "adjacency_monthly.parquet"
    if adj_path.exists():
        adj_df = pd.read_parquet(adj_path)
    else:
        adj_dfs = [pd.read_parquet(d / "adjacency.parquet")
                   for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]]
        adj_df = pd.concat(adj_dfs).drop_duplicates(subset=["period"]).sort_values("period")

    train_prices = train_df[NODES].values.astype(np.float32)
    val_prices   = val_df[NODES].values.astype(np.float32)
    test_prices  = test_df[NODES].values.astype(np.float32)

    train_dates  = train_df.index
    val_dates    = val_df.index
    test_dates   = test_df.index
    test_dates_window = test_dates[lookback:]

    # Ground truth for test
    _, y_true = sliding_windows(test_prices, lookback)

    # Load GDELT flat features
    print("\nLoading GDELT flat features...")
    gdelt_flat_df    = load_gdelt_flat(prices_df)
    gdelt_flat_train = gdelt_flat_df.loc[TRAIN_START:TRAIN_END].values.astype(np.float32)
    gdelt_flat_val   = gdelt_flat_df.loc[VAL_START:VAL_END].values.astype(np.float32)
    gdelt_flat_test  = gdelt_flat_df.loc[TEST_START:].values.astype(np.float32)
    print(f"  GDELT flat shape: train={gdelt_flat_train.shape}, "
          f"val={gdelt_flat_val.shape}, test={gdelt_flat_test.shape}")

    all_results = {}

    # -------------------------------------------------------------------------
    # GROUP 1 — Naive Baselines
    # -------------------------------------------------------------------------

    print("\n--- [1a] RandomWalk ---")
    rw      = RandomWalk()
    rw.fit(train_prices)
    rw_pred = rw.predict(test_prices, lookback=lookback)
    y_rw    = y_true[:len(rw_pred)]
    all_results["RandomWalk"] = evaluate_and_save("RandomWalk", y_rw, rw_pred, test_dates_window[:len(rw_pred)], lookback)
    print(f"  RMSE mean: {all_results['RandomWalk']['RMSE_mean']:.4f}")

    print("\n--- [1b] HistoricalMean ---")
    hm      = HistoricalMeanBaseline()
    hm.fit(train_prices)
    hm_pred = hm.predict(test_prices, lookback=lookback)
    all_results["HistoricalMean"] = evaluate_and_save("HistoricalMean", y_true[:len(hm_pred)], hm_pred, test_dates_window[:len(hm_pred)], lookback)
    print(f"  RMSE mean: {all_results['HistoricalMean']['RMSE_mean']:.4f}")

    if skip_arima:
        print("\n--- [1c] ARIMA --- SKIPPED (--skip-arima)")
        print("\n--- [1d] ARIMAX --- SKIPPED (--skip-arima)")
        # Load previously saved results if they exist
        for name in ["ARIMA", "ARIMAX"]:
            p = RESULTS_DIR / name / f"metrics_k{lookback}.json"
            if p.exists():
                with open(p) as f:
                    all_results[name] = json.load(f).get("overall", {})
    else:
        print("\n--- [1c] ARIMA ---")
        arima      = ARIMABaseline()
        arima.fit(train_prices)
        arima_pred = arima.predict(test_prices, lookback=lookback)
        all_results["ARIMA"] = evaluate_and_save("ARIMA", y_true[:len(arima_pred)], arima_pred, test_dates_window[:len(arima_pred)], lookback)
        print(f"  RMSE mean: {all_results['ARIMA']['RMSE_mean']:.4f}")

        print("\n--- [1d] ARIMAX ---")
        arimax      = ARIMAXBaseline()
        arimax.fit(train_prices, gdelt_flat_train=gdelt_flat_train)
        arimax_pred = arimax.predict(test_prices, lookback=lookback,
                                     gdelt_flat_test=gdelt_flat_test)
        all_results["ARIMAX"] = evaluate_and_save("ARIMAX", y_true[:len(arimax_pred)], arimax_pred, test_dates_window[:len(arimax_pred)], lookback)
        print(f"  RMSE mean: {all_results['ARIMAX']['RMSE_mean']:.4f}")

    # -------------------------------------------------------------------------
    # GROUP 2 — ML Baselines Without Graph Structure
    # -------------------------------------------------------------------------

    print("\n--- [2a] XGBoost ---")
    xgb      = XGBoostBaseline()
    xgb.fit(train_prices, lookback=lookback)
    xgb_pred = xgb.predict(test_prices, lookback=lookback)
    all_results["XGBoost"] = evaluate_and_save("XGBoost", y_true, xgb_pred, test_dates_window[:len(xgb_pred)], lookback)
    print(f"  RMSE mean: {all_results['XGBoost']['RMSE_mean']:.4f}")

    print("\n--- [2b] SVR ---")
    svr      = SVRBaseline()
    svr.fit(train_prices, lookback=lookback)
    svr_pred = svr.predict(test_prices, lookback=lookback)
    all_results["SVR"] = evaluate_and_save("SVR", y_true, svr_pred, test_dates_window[:len(svr_pred)], lookback)
    print(f"  RMSE mean: {all_results['SVR']['RMSE_mean']:.4f}")

    print("\n--- [2c] LSTM (prices only) ---")
    lstm      = LSTMBaseline()
    lstm.fit(train_prices, val_prices, lookback=lookback)
    lstm_pred = lstm.predict(test_prices, lookback=lookback)
    all_results["LSTM"] = evaluate_and_save("LSTM", y_true, lstm_pred, test_dates_window[:len(lstm_pred)], lookback)
    print(f"  RMSE mean: {all_results['LSTM']['RMSE_mean']:.4f}")

    print("\n--- [2d] LSTM + GDELT flat ---")
    lstm_gdelt = LSTMGDELTBaseline()
    lstm_gdelt.fit(train_prices, val_prices,
                   gdelt_flat_train, gdelt_flat_val,
                   lookback=lookback)
    lstm_gdelt_pred = lstm_gdelt.predict(test_prices, gdelt_flat_test,
                                         lookback=lookback)
    all_results["LSTM_GDELT"] = evaluate_and_save("LSTM_GDELT", y_true, lstm_gdelt_pred, test_dates_window[:len(lstm_gdelt_pred)], lookback)
    print(f"  RMSE mean: {all_results['LSTM_GDELT']['RMSE_mean']:.4f}")

    # -------------------------------------------------------------------------
    # GROUP 3 — Graph Baselines Without Dynamic Edges
    # -------------------------------------------------------------------------

    print("\n--- [3a] GCN + static adjacency ---")
    gcn_static      = GCNStaticBaseline()
    gcn_static.fit(prices_df, adj_df, lookback=lookback)
    gcn_static_pred = gcn_static.predict(prices_df, adj_df, lookback=lookback)
    all_results["GCNStatic"] = evaluate_and_save("GCNStatic", y_true[:len(gcn_static_pred)], gcn_static_pred, test_dates_window[:len(gcn_static_pred)], lookback)
    print(f"  RMSE mean: {all_results['GCNStatic']['RMSE_mean']:.4f}")

    print("\n--- [3b] GAT + static adjacency ---")
    gat_static      = GATStaticBaseline()
    gat_static.fit(prices_df, adj_df, lookback=lookback)
    gat_static_pred = gat_static.predict(prices_df, adj_df, lookback=lookback)
    all_results["GATStatic"] = evaluate_and_save("GATStatic", y_true[:len(gat_static_pred)], gat_static_pred, test_dates_window[:len(gat_static_pred)], lookback)
    print(f"  RMSE mean: {all_results['GATStatic']['RMSE_mean']:.4f}")

    print("\n--- [3c] DCRNN ---")
    dcrnn      = DCRNNBaseline()
    dcrnn.fit(prices_df, adj_df, lookback=lookback)
    dcrnn_pred = dcrnn.predict(prices_df, adj_df, lookback=lookback)
    all_results["DCRNN"] = evaluate_and_save("DCRNN", y_true[:len(dcrnn_pred)], dcrnn_pred, test_dates_window[:len(dcrnn_pred)], lookback)
    print(f"  RMSE mean: {all_results['DCRNN']['RMSE_mean']:.4f}")

    # -------------------------------------------------------------------------
    # GROUP 4 — Strong Temporal Baselines
    # -------------------------------------------------------------------------

    print("\n--- [4a] TFT ---")
    tft      = TFTBaseline()
    tft.fit(train_prices, val_prices, lookback=lookback)
    tft_pred = tft.predict(test_prices, lookback=lookback)
    all_results["TFT"] = evaluate_and_save("TFT", y_true, tft_pred, test_dates_window[:len(tft_pred)], lookback)
    print(f"  RMSE mean: {all_results['TFT']['RMSE_mean']:.4f}")

    print("\n--- [4b] PatchTST ---")
    patchtst      = PatchTSTBaseline()
    patchtst.fit(train_prices, val_prices, lookback=lookback)
    patchtst_pred = patchtst.predict(test_prices, lookback=lookback)
    all_results["PatchTST"] = evaluate_and_save("PatchTST", y_true, patchtst_pred, test_dates_window[:len(patchtst_pred)], lookback)
    print(f"  RMSE mean: {all_results['PatchTST']['RMSE_mean']:.4f}")

    print("\n--- [4c] N-BEATS ---")
    nbeats      = NBEATSBaseline()
    nbeats.fit(train_prices, val_prices, lookback=lookback)
    nbeats_pred = nbeats.predict(test_prices, lookback=lookback)
    all_results["N-BEATS"] = evaluate_and_save("N-BEATS", y_true, nbeats_pred, test_dates_window[:len(nbeats_pred)], lookback)
    print(f"  RMSE mean: {all_results['N-BEATS']['RMSE_mean']:.4f}")

    # -------------------------------------------------------------------------
    # GROUP 5 — Domain-Specific Prior Work
    # -------------------------------------------------------------------------

    print("\n--- [5a] EMD-LSTM ---")
    emd_lstm      = EMDLSTMBaseline()
    emd_lstm.fit(train_prices, val_prices, lookback=lookback)
    emd_lstm_pred = emd_lstm.predict(test_prices, lookback=lookback)
    all_results["EMD-LSTM"] = evaluate_and_save("EMD-LSTM", y_true, emd_lstm_pred, test_dates_window[:len(emd_lstm_pred)], lookback)
    print(f"  RMSE mean: {all_results['EMD-LSTM']['RMSE_mean']:.4f}")

    print("\n--- [5b] SentimentLSTM ---")
    sent_lstm      = SentimentLSTMBaseline()
    sent_lstm.fit(train_prices, val_prices,
                  gdelt_flat_train, gdelt_flat_val,
                  lookback=lookback)
    sent_lstm_pred = sent_lstm.predict(test_prices, gdelt_flat_test,
                                       lookback=lookback)
    all_results["SentimentLSTM"] = evaluate_and_save("SentimentLSTM", y_true, sent_lstm_pred, test_dates_window[:len(sent_lstm_pred)], lookback)
    print(f"  RMSE mean: {all_results['SentimentLSTM']['RMSE_mean']:.4f}")

    print("\n--- [5c] GPR-LSTM ---")
    gpr_lstm      = GPRLSTMBaseline()
    gpr_lstm.fit(train_prices, val_prices,
                 gdelt_flat_train, gdelt_flat_val,
                 train_dates=train_dates, val_dates=val_dates,
                 lookback=lookback)
    gpr_lstm_pred = gpr_lstm.predict(test_prices, gdelt_flat_test,
                                     test_dates=test_dates,
                                     lookback=lookback)
    all_results["GPR-LSTM"] = evaluate_and_save("GPR-LSTM", y_true, gpr_lstm_pred, test_dates_window[:len(gpr_lstm_pred)], lookback)
    print(f"  RMSE mean: {all_results['GPR-LSTM']['RMSE_mean']:.4f}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"baselines_k{lookback}.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved -> {out}")

    print_comparison_table(all_results)
    return all_results


def print_comparison_table(results: dict):
    print()
    print("=" * 85)
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'MAPE%':>8} {'R²':>8} {'DA%':>8}")
    print("-" * 20 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8 + " " + "-" * 8)
    groups = [
        ("-- Group 1: Naive --", ["RandomWalk", "HistoricalMean", "ARIMA", "ARIMAX"]),
        ("-- Group 2: ML/No Graph --", ["XGBoost", "SVR", "LSTM", "LSTM_GDELT"]),
        ("-- Group 3: Graph/Static --", ["GCNStatic", "GATStatic", "DCRNN"]),
        ("-- Group 4: Strong Temporal --", ["TFT", "PatchTST", "N-BEATS"]),
        ("-- Group 5: Domain Prior --", ["EMD-LSTM", "SentimentLSTM", "GPR-LSTM"]),
    ]
    for group_label, keys in groups:
        print(f"\n{group_label}")
        for k in keys:
            if k in results:
                r = results[k]
                r2_val = r.get('R2_mean', 0.0)
                da_val = r.get('DA_mean', 0.0)
                print(f"  {k:<18} {r['RMSE_mean']:>8.4f} "
                      f"{r['MAE_mean']:>8.4f} {r['MAPE_mean']:>8.4f} "
                      f"{r2_val:>8.4f} {da_val:>8.2f}")
    print("=" * 85)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lookback",    type=int, default=LOOKBACK_WINDOW)
    parser.add_argument("--skip-arima",  action="store_true",
                        help="Skip ARIMA and ARIMAX (use if already computed)")
    args = parser.parse_args()
    run_all_baselines(lookback=args.lookback, skip_arima=args.skip_arima)
