"""
build_dataset.py
Merges price data, GDELT tensor, and Comtrade adjacency into model-ready tensors.

Outputs (saved to data/):
  aligned_prices.parquet   — [N_days × 5] daily closing prices, all gaps forward-filled
  dataset_samples.parquet  — one row per training sample with all inputs
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    PRICE_FILES, NODES, N_NODES, ADJACENCY_FILE, GDELT_TENSOR_FILE,
    TRAIN_START, TRAIN_END, VAL_START, VAL_END, TEST_START, DATA_DIR,
    LOOKBACK_WINDOW
)


def load_prices() -> pd.DataFrame:
    """Load all 5 price CSVs, align on trading days, forward-fill gaps."""
    dfs = {}
    for name, path in PRICE_FILES.items():
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        df.columns = [name]
        dfs[name] = df

    prices = pd.concat(dfs.values(), axis=1)
    prices.columns = NODES
    prices = prices.sort_index()

    # Forward-fill up to 5 days (weekends, holidays)
    prices = prices.ffill(limit=5)

    # Keep only days where at least WTI and Brent are available
    prices = prices.dropna(subset=["WTI", "Brent"])

    print(f"Prices: {len(prices)} trading days | {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Missing values per column:\n{prices.isna().sum()}")
    return prices


def load_adjacency() -> pd.DataFrame:
    """Load monthly 5x5 Comtrade adjacency matrices."""
    if not ADJACENCY_FILE.exists():
        raise FileNotFoundError(
            f"{ADJACENCY_FILE} not found. Run comtrade parser first."
        )
    adj = pd.read_parquet(ADJACENCY_FILE)
    print(f"Adjacency: {len(adj)} months | {adj['period'].min()} to {adj['period'].max()}")
    return adj


def get_adjacency_for_date(adj_df: pd.DataFrame, date: pd.Timestamp) -> np.ndarray:
    """Return the 5x5 adjacency matrix for the month containing `date`."""
    period = date.year * 100 + date.month
    row = adj_df[adj_df["period"] == period]
    if row.empty:
        # Fall back to most recent available month
        past = adj_df[adj_df["period"] < period]
        if past.empty:
            return np.eye(N_NODES) / N_NODES
        row = past.iloc[[-1]]

    mat_cols = [c for c in row.columns if c.startswith("col_")]
    return row[mat_cols].values.reshape(N_NODES, N_NODES)


def load_gdelt() -> pd.DataFrame:
    """
    Load GDELT daily tensor.

    The parquet has one row per (date, from_node, to_node) with columns
    [date, from_node, to_node, GoldsteinScale, AvgTone, NumMentions].
    Returns a DataFrame indexed by date with 75 columns (5×5×3 flattened,
    row-major over (from_node, to_node, channel), diagonal zeros).
    """
    if not GDELT_TENSOR_FILE.exists():
        raise FileNotFoundError(
            f"{GDELT_TENSOR_FILE} not found. Run collect_gdelt.py first."
        )
    raw = pd.read_parquet(GDELT_TENSOR_FILE)
    raw.columns = [c.lower() if c.lower() == 'date' else c for c in raw.columns]
    if 'Date' in raw.columns:
        raw = raw.rename(columns={'Date': 'date'})
    raw['date'] = pd.to_datetime(raw['date'])

    # Drop self-loops
    raw = raw[raw['from_node'] != raw['to_node']].copy()

    # Map unique dates to integer row indices
    dates = sorted(raw['date'].unique())
    date_to_idx = {d: i for i, d in enumerate(dates)}
    row_idx = raw['date'].map(date_to_idx).values

    # Compute flat column index for each channel:  i*N*3 + j*3 + ch
    fi = raw['from_node'].values.astype(int)
    ti = raw['to_node'].values.astype(int)
    base = fi * N_NODES * 3 + ti * 3

    arr = np.zeros((len(dates), N_NODES * N_NODES * 3), dtype=np.float32)
    arr[row_idx, base + 0] = raw['GoldsteinScale'].values
    arr[row_idx, base + 1] = raw['AvgTone'].values
    arr[row_idx, base + 2] = raw['NumMentions'].values

    gdelt = pd.DataFrame(arr, index=pd.DatetimeIndex(dates))
    gdelt = gdelt.sort_index()
    print(f"GDELT: {len(gdelt)} days | {gdelt.index[0].date()} to {gdelt.index[-1].date()}")
    return gdelt


def build_aligned_prices():
    """Step 1: Save aligned price DataFrame."""
    prices = load_prices()
    out = DATA_DIR / "aligned_prices.parquet"
    prices.to_parquet(out)
    print(f"Saved: {out}")
    return prices


def build_samples(lookback: int = LOOKBACK_WINDOW):
    """
    Step 2: Build training samples.
    Each sample: (prices_window, gdelt_today, adjacency_today, target_prices)
    """
    prices = pd.read_parquet(DATA_DIR / "aligned_prices.parquet")
    # Restrict to model window (all 5 series are complete from TRAIN_START)
    prices = prices.loc[TRAIN_START:].dropna()
    adj_df = load_adjacency()
    gdelt  = load_gdelt()

    # Align all on common trading days
    common_dates = prices.index.intersection(gdelt.index)
    prices = prices.loc[common_dates]
    gdelt  = gdelt.loc[common_dates]

    records = []
    dates = prices.index.tolist()

    for i in range(lookback, len(dates) - 1):
        t      = dates[i]
        t_next = dates[i + 1]
        window = prices.iloc[i - lookback: i].values   # [k × 5]
        target = prices.loc[t_next].values              # [5]
        gdelt_t = gdelt.loc[t].values                   # [75] = 5×5×3 flattened
        adj_t   = get_adjacency_for_date(adj_df, t)    # [5×5]

        records.append({
            "date":      t,
            "prices_window":  window.tolist(),
            "gdelt_tensor":   gdelt_t.tolist(),
            "adjacency":      adj_t.tolist(),
            "target":         target.tolist(),
        })

    df = pd.DataFrame(records)
    out = DATA_DIR / f"dataset_k{lookback}.parquet"
    df.to_parquet(out)
    print(f"Saved {len(df)} samples -> {out}")
    return df


if __name__ == "__main__":
    print("=== Step 1: Align prices ===")
    build_aligned_prices()

    print("\n=== Step 2: Build samples (requires GDELT + Comtrade) ===")
    try:
        for k in [10, 20, 30]:
            build_samples(lookback=k)
    except FileNotFoundError as e:
        print(f"Skipped: {e}")
