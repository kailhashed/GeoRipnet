"""
collect_gdelt.py
Downloads GDELT event data and builds daily 5×5×3 tensors.

Node map:
  0=WTI/USA, 1=Brent/GBR+NOR, 2=OPEC/SAU, 3=Urals/RUS, 4=Indian/IND

Output:
  data/gdelt_data/daily_gdelt_tensor.parquet
  Columns: date, from_node, to_node, GoldsteinScale, AvgTone, NumMentions

Usage:
  python src/collect_gdelt.py
  python src/collect_gdelt.py --start 2015-01-01 --end 2020-12-31
"""
import sys
import os
import io
import zipfile
import argparse
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import product

# Allow importing config from same directory
sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, GDELT_DIR, RAW_CACHE_DIR, GDELT_TENSOR_FILE

# ── Constants ──────────────────────────────────────────────────────────────────

FIPS_TO_NODE = {'US': 0, 'UK': 1, 'NO': 1, 'SA': 2, 'RS': 3, 'IN': 4}
N_NODES = 5

# CAMEO codes: hostile/conflict (startswith any of these prefixes)
CAMEO_PREFIXES = {'13', '17', '18', '19', '20'}

# All 20 non-diagonal (from, to) pairs
ALL_PAIRS = [(i, j) for i, j in product(range(N_NODES), range(N_NODES)) if i != j]

# GDELT 1.0 column positions (0-indexed, tab-separated, NO header)
V1_COLS = {
    'SQLDATE': 1,
    'Actor1CountryCode': 5,
    'Actor2CountryCode': 15,
    'EventCode': 26,
    'GoldsteinScale': 30,
    'NumMentions': 31,
    'AvgTone': 34,
}
V1_USE_COLS = list(V1_COLS.values())
V1_COL_NAMES = list(V1_COLS.keys())

# GDELT 2.0 uses the same column names but has a header row
V2_USE_COLS = list(V1_COLS.keys())

REQUEST_TIMEOUT = 120  # seconds


# ── Helpers ────────────────────────────────────────────────────────────────────

def cameo_filter(event_codes: pd.Series) -> pd.Series:
    """Return boolean mask for rows matching our CAMEO prefixes."""
    ec = event_codes.astype(str).str.strip()
    mask = pd.Series(False, index=event_codes.index)
    for prefix in CAMEO_PREFIXES:
        mask |= ec.str.startswith(prefix)
    return mask


def fips_to_node(code: str) -> int:
    """Map FIPS country code to node index, or -1 if not in our set."""
    return FIPS_TO_NODE.get(str(code).strip().upper(), -1)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a filtered GDELT DataFrame with columns
    [date_str, from_node, to_node, GoldsteinScale, AvgTone, NumMentions],
    aggregate to one row per (date, from_node, to_node).
    """
    grp = df.groupby(['date', 'from_node', 'to_node'], as_index=False).agg(
        GoldsteinScale=('GoldsteinScale', 'mean'),
        AvgTone=('AvgTone', 'mean'),
        NumMentions=('NumMentions', 'sum'),
    )
    return grp


def fill_all_pairs(grp: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """
    Ensure all 20 non-diagonal (from, to) pairs exist for a given date.
    Missing pairs are filled with zeros.
    """
    full = pd.DataFrame(ALL_PAIRS, columns=['from_node', 'to_node'])
    full['date'] = date_str
    merged = full.merge(grp, on=['date', 'from_node', 'to_node'], how='left')
    merged[['GoldsteinScale', 'AvgTone', 'NumMentions']] = (
        merged[['GoldsteinScale', 'AvgTone', 'NumMentions']].fillna(0.0)
    )
    return merged[['date', 'from_node', 'to_node', 'GoldsteinScale', 'AvgTone', 'NumMentions']]


def parse_gdelt_df(raw_df: pd.DataFrame, is_v1: bool) -> pd.DataFrame:
    """
    Given a raw GDELT DataFrame, filter and return a cleaned DataFrame
    with columns [date, from_node, to_node, GoldsteinScale, AvgTone, NumMentions].
    """
    if is_v1:
        raw_df.columns = list(range(len(raw_df.columns)))
        # Select only needed columns
        try:
            df = raw_df[V1_USE_COLS].copy()
        except KeyError:
            # Some v1 files may have fewer columns — skip gracefully
            return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                         'GoldsteinScale', 'AvgTone', 'NumMentions'])
        df.columns = V1_COL_NAMES
    else:
        # v2 has header; keep only the columns we need
        needed = [c for c in V2_USE_COLS if c in raw_df.columns]
        if len(needed) < len(V2_USE_COLS):
            return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                         'GoldsteinScale', 'AvgTone', 'NumMentions'])
        df = raw_df[V2_USE_COLS].copy()

    # Coerce numeric
    df['SQLDATE'] = df['SQLDATE'].astype(str).str.strip()
    df['GoldsteinScale'] = pd.to_numeric(df['GoldsteinScale'], errors='coerce')
    df['AvgTone'] = pd.to_numeric(df['AvgTone'], errors='coerce')
    df['NumMentions'] = pd.to_numeric(df['NumMentions'], errors='coerce').fillna(0)

    # Filter CAMEO
    df = df[cameo_filter(df['EventCode'])].copy()
    if df.empty:
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    # Map country codes to nodes
    df['from_node'] = df['Actor1CountryCode'].apply(fips_to_node)
    df['to_node']   = df['Actor2CountryCode'].apply(fips_to_node)

    # Keep only rows where both actors are in our node set and are different
    df = df[(df['from_node'] >= 0) & (df['to_node'] >= 0) & (df['from_node'] != df['to_node'])].copy()
    if df.empty:
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    # Parse date as YYYYMMDD string → YYYY-MM-DD
    df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d', errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date', 'GoldsteinScale', 'AvgTone'])

    return df[['date', 'from_node', 'to_node', 'GoldsteinScale', 'AvgTone', 'NumMentions']].copy()


def download_zip(url: str, timeout: int = REQUEST_TIMEOUT) -> bytes:
    """Download a zip file and return its raw bytes."""
    resp = requests.get(url, timeout=timeout, stream=True)
    resp.raise_for_status()
    return resp.content


def read_gdelt_zip(zip_bytes: bytes, is_v1: bool) -> pd.DataFrame:
    """Read GDELT CSV from zip bytes into a DataFrame."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        csv_names = [n for n in z.namelist() if n.endswith('.CSV') or n.endswith('.csv')]
        if not csv_names:
            return pd.DataFrame()
        with z.open(csv_names[0]) as f:
            if is_v1:
                df = pd.read_csv(f, sep='\t', header=None, dtype=str,
                                 on_bad_lines='skip', low_memory=False)
            else:
                df = pd.read_csv(f, sep='\t', header=0, dtype=str,
                                 on_bad_lines='skip', low_memory=False)
    return df


# ── Year-level (GDELT 1.0, 2010-2014) ────────────────────────────────────────

def process_v1_year(year: int, trading_dates: set) -> pd.DataFrame:
    """
    Download the full annual GDELT 1.0 zip for `year`, filter to our trading
    dates and node pairs, and return aggregated daily rows.
    """
    url = f"http://data.gdeltproject.org/events/{year}.zip"
    print(f"  Downloading GDELT 1.0 — {year} from {url} ...")
    try:
        zip_bytes = download_zip(url)
    except Exception as e:
        print(f"  WARNING: Could not download {year}: {e}")
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    raw_df = read_gdelt_zip(zip_bytes, is_v1=True)
    if raw_df.empty:
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    parsed = parse_gdelt_df(raw_df, is_v1=True)
    if parsed.empty:
        return parsed

    # Keep only trading dates
    parsed = parsed[parsed['date'].isin(trading_dates)]
    if parsed.empty:
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    # Aggregate to daily
    agg = aggregate_daily(parsed)

    # Fill all pairs per date
    date_frames = []
    for d in agg['date'].unique():
        d_rows = agg[agg['date'] == d]
        date_frames.append(fill_all_pairs(d_rows, d))

    return pd.concat(date_frames, ignore_index=True) if date_frames else pd.DataFrame(
        columns=['date', 'from_node', 'to_node', 'GoldsteinScale', 'AvgTone', 'NumMentions']
    )


# ── Day-level (GDELT 2.0, 2015+) ─────────────────────────────────────────────

def process_v2_day(date_str: str) -> pd.DataFrame:
    """
    Download a single GDELT 2.0 daily export, parse, and return aggregated rows
    for all 20 (from, to) pairs for that date.
    """
    date_compact = date_str.replace('-', '')  # YYYYMMDD
    url = f"http://data.gdeltproject.org/gdeltv2/{date_compact}.export.CSV.zip"

    try:
        zip_bytes = download_zip(url)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            # File just doesn't exist for this date — fill zeros
            pass
        else:
            print(f"  WARNING: HTTP error for {date_str}: {e}")
        return fill_all_pairs(pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                                     'GoldsteinScale', 'AvgTone', 'NumMentions']),
                               date_str)
    except Exception as e:
        print(f"  WARNING: Could not download {date_str}: {e}")
        return fill_all_pairs(pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                                     'GoldsteinScale', 'AvgTone', 'NumMentions']),
                               date_str)

    raw_df = read_gdelt_zip(zip_bytes, is_v1=False)
    if raw_df.empty:
        return fill_all_pairs(pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                                     'GoldsteinScale', 'AvgTone', 'NumMentions']),
                               date_str)

    parsed = parse_gdelt_df(raw_df, is_v1=False)
    parsed = parsed[parsed['date'] == date_str]

    agg = aggregate_daily(parsed) if not parsed.empty else pd.DataFrame(
        columns=['date', 'from_node', 'to_node', 'GoldsteinScale', 'AvgTone', 'NumMentions']
    )
    return fill_all_pairs(agg, date_str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(start_override: str = None, end_override: str = None):
    GDELT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load trading dates
    prices = pd.read_parquet(DATA_DIR / "aligned_prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    all_dates = sorted(prices.index.strftime('%Y-%m-%d').tolist())

    if start_override:
        all_dates = [d for d in all_dates if d >= start_override]
    if end_override:
        all_dates = [d for d in all_dates if d <= end_override]

    print(f"Trading dates to process: {len(all_dates)} ({all_dates[0]} to {all_dates[-1]})")

    # Load already-processed dates (resumable)
    already_done = set()
    if GDELT_TENSOR_FILE.exists():
        existing = pd.read_parquet(GDELT_TENSOR_FILE)
        already_done = set(existing['date'].unique())
        print(f"Already processed: {len(already_done)} dates — will skip those")
    else:
        existing = pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                          'GoldsteinScale', 'AvgTone', 'NumMentions'])

    trading_date_set = set(all_dates)
    all_results = [existing] if not existing.empty else []

    # ── GDELT 1.0: 2010–2014 ─────────────────────────────────────────────────
    v1_years_needed = []
    for year in range(2010, 2015):
        year_dates = {d for d in all_dates if d.startswith(str(year)) and d not in already_done}
        if year_dates:
            v1_years_needed.append((year, year_dates))

    for year, year_dates in v1_years_needed:
        print(f"\n--- Processing GDELT 1.0 year {year} ({len(year_dates)} trading days needed) ---")
        year_df = process_v1_year(year, year_dates)
        if not year_df.empty:
            all_results.append(year_df)
            # Save incrementally
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_parquet(GDELT_TENSOR_FILE, index=False)
            already_done.update(year_df['date'].unique())
            print(f"  Saved {len(year_df)} rows for year {year}")

    # ── GDELT 2.0: 2015+ ──────────────────────────────────────────────────────
    v2_dates = [d for d in all_dates if d >= '2015-01-01' and d not in already_done]
    print(f"\n--- Processing GDELT 2.0 daily ({len(v2_dates)} dates) ---")

    for idx, date_str in enumerate(v2_dates):
        if idx > 0 and idx % 100 == 0:
            combined = pd.concat(all_results, ignore_index=True)
            combined.to_parquet(GDELT_TENSOR_FILE, index=False)
            print(f"  Progress: {idx}/{len(v2_dates)} | Saved checkpoint")

        day_df = process_v2_day(date_str)
        all_results.append(day_df)

    # Final save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        # Deduplicate in case of overlap
        combined = combined.drop_duplicates(subset=['date', 'from_node', 'to_node'], keep='last')
        combined = combined.sort_values(['date', 'from_node', 'to_node']).reset_index(drop=True)
        combined.to_parquet(GDELT_TENSOR_FILE, index=False)
        print(f"\nDone! Saved {len(combined)} rows to {GDELT_TENSOR_FILE}")
        print(combined.head(10))
    else:
        print("No data collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect GDELT data and build daily tensors")
    parser.add_argument("--start", type=str, default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",   type=str, default=None, help="End date YYYY-MM-DD")
    args = parser.parse_args()
    main(start_override=args.start, end_override=args.end)
