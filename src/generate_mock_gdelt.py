"""
generate_mock_gdelt.py
Creates a realistic mock GDELT tensor for pipeline testing.

Generates daily 5×5×3 tensors for all trading dates in aligned_prices.parquet.
The RUS→IND pair (3,4) gets elevated NumMentions after 2022-02-24.

Output: data/gdelt_data/daily_gdelt_tensor.parquet
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, GDELT_DIR, GDELT_TENSOR_FILE

SEED = 42
N_NODES = 5
ALL_PAIRS = [(i, j) for i, j in product(range(N_NODES), range(N_NODES)) if i != j]
UKRAINE_WAR_START = '2022-02-24'


def main():
    GDELT_DIR.mkdir(parents=True, exist_ok=True)

    # Load trading dates
    prices = pd.read_parquet(DATA_DIR / "aligned_prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    dates = sorted(prices.index.strftime('%Y-%m-%d').tolist())
    print(f"Loaded {len(dates)} trading dates: {dates[0]} to {dates[-1]}")

    rng = np.random.default_rng(SEED)
    records = []

    for date_str in dates:
        for (from_node, to_node) in ALL_PAIRS:
            goldstein  = float(rng.uniform(-5.0, 0.0))
            avg_tone   = float(rng.uniform(-3.0, 3.0))

            # RUS→IND elevated after Ukraine war start
            if from_node == 3 and to_node == 4 and date_str >= UKRAINE_WAR_START:
                num_mentions = int(rng.integers(5, 201))
            else:
                num_mentions = int(rng.integers(0, 51))

            records.append({
                'date':           date_str,
                'from_node':      from_node,
                'to_node':        to_node,
                'GoldsteinScale': goldstein,
                'AvgTone':        avg_tone,
                'NumMentions':    num_mentions,
            })

    df = pd.DataFrame(records)
    df = df.sort_values(['date', 'from_node', 'to_node']).reset_index(drop=True)
    df.to_parquet(GDELT_TENSOR_FILE, index=False)

    n_dates = df['date'].nunique()
    n_rows  = len(df)
    print(f"\nMock GDELT tensor saved to {GDELT_TENSOR_FILE}")
    print(f"  Dates: {n_dates}  |  Rows: {n_rows}  |  Rows/date: {n_rows / n_dates:.0f}")
    print(f"\nSample rows:")
    print(df.head(20).to_string(index=False))

    # Spot-check RUS->IND after war
    post_war = df[(df['from_node'] == 3) & (df['to_node'] == 4) & (df['date'] >= UKRAINE_WAR_START)]
    pre_war  = df[(df['from_node'] == 3) & (df['to_node'] == 4) & (df['date'] < UKRAINE_WAR_START)]
    print(f"\nRUS→IND NumMentions — pre-war mean:  {pre_war['NumMentions'].mean():.1f}")
    print(f"RUS→IND NumMentions — post-war mean: {post_war['NumMentions'].mean():.1f}")


if __name__ == "__main__":
    main()
