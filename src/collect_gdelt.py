"""
collect_gdelt.py
Downloads GDELT event data and builds daily 5x5x3 tensors.

Node map:
  0=WTI/USA, 1=Brent/GBR+NOR, 2=OPEC/SAU, 3=ESPO/RUS, 4=Indian/IND

URL format (confirmed working):
  http://data.gdeltproject.org/events/YYYYMMDD.export.CSV.zip
  Available from 2013-04-01 onwards. Dates before that get zeros.

File format: tab-separated, NO header row, 58 columns.
  Col  1 = SQLDATE (YYYYMMDD)
  Col  7 = Actor1CountryCode (3-letter CAMEO: USA, GBR, RUS, ...)
  Col 17 = Actor2CountryCode
  Col 26 = EventCode (CAMEO event code)
  Col 30 = GoldsteinScale
  Col 31 = NumMentions
  Col 34 = AvgTone

Output:
  data/gdelt_data/daily_gdelt_tensor.parquet
  Columns: date, from_node, to_node, GoldsteinScale, AvgTone, NumMentions

Usage:
  python src/collect_gdelt.py
  python src/collect_gdelt.py --start 2013-04-01 --end 2020-12-31
"""
import sys
import io
import zipfile
import argparse
import requests
import pandas as pd
from pathlib import Path
from itertools import product

sys.path.insert(0, str(Path(__file__).parent))
from config import DATA_DIR, GDELT_DIR, RAW_CACHE_DIR, GDELT_TENSOR_FILE

# ── Constants ──────────────────────────────────────────────────────────────────

# CAMEO 3-letter country codes used by GDELT (not FIPS 2-letter)
CAMEO_TO_NODE = {'USA': 0, 'GBR': 1, 'NOR': 1, 'SAU': 2, 'RUS': 3, 'IND': 4}
N_NODES = 5

# CAMEO event codes: hostile/conflict events
CAMEO_PREFIXES = {'13', '17', '18', '19', '20'}

# All 20 non-diagonal (from, to) pairs
ALL_PAIRS = [(i, j) for i, j in product(range(N_NODES), range(N_NODES)) if i != j]

# Column positions (0-indexed, no header row)
COL_SQLDATE           = 1
COL_ACTOR1_COUNTRY    = 7
COL_ACTOR2_COUNTRY    = 17
COL_EVENTCODE         = 26
COL_GOLDSTEIN         = 30
COL_NUMMENTIONS       = 31
COL_AVGTONE           = 34

USE_COLS  = [COL_SQLDATE, COL_ACTOR1_COUNTRY, COL_ACTOR2_COUNTRY,
             COL_EVENTCODE, COL_GOLDSTEIN, COL_NUMMENTIONS, COL_AVGTONE]
COL_NAMES = ['SQLDATE', 'Actor1CountryCode', 'Actor2CountryCode',
             'EventCode', 'GoldsteinScale', 'NumMentions', 'AvgTone']

# Earliest date with a file at the /events/ URL
GDELT_EARLIEST = '2013-04-01'

REQUEST_TIMEOUT = 120


# ── Helpers ────────────────────────────────────────────────────────────────────

def cameo_filter(event_codes: pd.Series) -> pd.Series:
    ec = event_codes.astype(str).str.strip()
    mask = pd.Series(False, index=event_codes.index)
    for prefix in CAMEO_PREFIXES:
        mask |= ec.str.startswith(prefix)
    return mask


def country_to_node(code: str) -> int:
    return CAMEO_TO_NODE.get(str(code).strip().upper(), -1)


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(['date', 'from_node', 'to_node'], as_index=False).agg(
        GoldsteinScale=('GoldsteinScale', 'mean'),
        AvgTone=('AvgTone', 'mean'),
        NumMentions=('NumMentions', 'sum'),
    )


def fill_all_pairs(grp: pd.DataFrame, date_str: str) -> pd.DataFrame:
    full = pd.DataFrame(ALL_PAIRS, columns=['from_node', 'to_node'])
    full['date'] = date_str
    merged = full.merge(grp, on=['date', 'from_node', 'to_node'], how='left')
    merged[['GoldsteinScale', 'AvgTone', 'NumMentions']] = (
        merged[['GoldsteinScale', 'AvgTone', 'NumMentions']].fillna(0.0)
    )
    return merged[['date', 'from_node', 'to_node', 'GoldsteinScale', 'AvgTone', 'NumMentions']]


def parse_gdelt_raw(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Parse raw GDELT DataFrame (no header) into filtered event rows."""
    raw_df.columns = list(range(len(raw_df.columns)))
    if len(raw_df.columns) <= max(USE_COLS):
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])
    df = raw_df[USE_COLS].copy()
    df.columns = COL_NAMES

    df['SQLDATE']       = df['SQLDATE'].astype(str).str.strip()
    df['GoldsteinScale'] = pd.to_numeric(df['GoldsteinScale'], errors='coerce')
    df['AvgTone']        = pd.to_numeric(df['AvgTone'],        errors='coerce')
    df['NumMentions']    = pd.to_numeric(df['NumMentions'],    errors='coerce').fillna(0)

    # Keep only hostile/conflict events
    df = df[cameo_filter(df['EventCode'])].copy()
    if df.empty:
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    # Map country codes to nodes
    df['from_node'] = df['Actor1CountryCode'].apply(country_to_node)
    df['to_node']   = df['Actor2CountryCode'].apply(country_to_node)

    # Keep only rows where both actors are in our node set and differ
    df = df[(df['from_node'] >= 0) & (df['to_node'] >= 0) &
            (df['from_node'] != df['to_node'])].copy()
    if df.empty:
        return pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                     'GoldsteinScale', 'AvgTone', 'NumMentions'])

    df['date'] = pd.to_datetime(df['SQLDATE'], format='%Y%m%d',
                                errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date', 'GoldsteinScale', 'AvgTone'])
    return df[['date', 'from_node', 'to_node',
               'GoldsteinScale', 'AvgTone', 'NumMentions']].copy()


def download_and_parse_day(date_str: str) -> pd.DataFrame:
    """
    Download GDELT zip for `date_str` (YYYY-MM-DD), parse, filter to
    events on that exact date, and return fill_all_pairs result.
    """
    date_compact = date_str.replace('-', '')
    url = f"http://data.gdeltproject.org/events/{date_compact}.export.CSV.zip"
    empty = pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                   'GoldsteinScale', 'AvgTone', 'NumMentions'])
    try:
        resp = requests.get(url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        zip_bytes = resp.content
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            pass  # no file for this date — fill zeros
        else:
            print(f"  WARNING HTTP {date_str}: {e}")
        return fill_all_pairs(empty, date_str)
    except Exception as e:
        print(f"  WARNING {date_str}: {e}")
        return fill_all_pairs(empty, date_str)

    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
            csv_names = [n for n in z.namelist()
                         if n.upper().endswith('.CSV')]
            if not csv_names:
                return fill_all_pairs(empty, date_str)
            with z.open(csv_names[0]) as f:
                raw_df = pd.read_csv(f, sep='\t', header=None, dtype=str,
                                     on_bad_lines='skip', low_memory=False)
    except Exception as e:
        print(f"  WARNING unzip {date_str}: {e}")
        return fill_all_pairs(empty, date_str)

    parsed = parse_gdelt_raw(raw_df)
    # Filter to only events whose SQLDATE matches the file date
    parsed = parsed[parsed['date'] == date_str]

    agg = aggregate_daily(parsed) if not parsed.empty else empty
    return fill_all_pairs(agg, date_str)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(start_override: str = None, end_override: str = None):
    GDELT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Load trading dates from aligned prices
    prices = pd.read_parquet(DATA_DIR / "aligned_prices.parquet")
    prices.index = pd.to_datetime(prices.index)
    all_dates = sorted(prices.index.strftime('%Y-%m-%d').tolist())

    if start_override:
        all_dates = [d for d in all_dates if d >= start_override]
    if end_override:
        all_dates = [d for d in all_dates if d <= end_override]

    print(f"Trading dates to process: {len(all_dates)} "
          f"({all_dates[0]} to {all_dates[-1]})")
    print(f"GDELT files available from {GDELT_EARLIEST} — "
          f"earlier dates will be zero-filled.")

    # Resumable: skip already-done dates
    already_done = set()
    if GDELT_TENSOR_FILE.exists():
        existing = pd.read_parquet(GDELT_TENSOR_FILE)
        # Only keep dates that have at least one non-zero row (real signal)
        # Zero-only dates from the broken previous run should be re-processed
        nonzero_dates = set(
            existing.loc[existing['NumMentions'] > 0, 'date'].unique()
        )
        zero_only_dates = set(existing['date'].unique()) - nonzero_dates
        if zero_only_dates:
            print(f"  Dropping {len(zero_only_dates)} zero-only dates "
                  f"(from previous broken run) — will re-download")
            existing = existing[existing['date'].isin(nonzero_dates)]
        already_done = set(existing['date'].unique())
        print(f"Already have signal for: {len(already_done)} dates")
    else:
        existing = pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                          'GoldsteinScale', 'AvgTone', 'NumMentions'])

    all_results = [existing] if not existing.empty else []

    # Dates to process: skip already done AND skip pre-2013-04-01 (no files)
    dates_to_do = [
        d for d in all_dates
        if d not in already_done and d >= GDELT_EARLIEST
    ]
    # Pre-GDELT dates: zero-fill once if not already present
    pre_gdelt = [
        d for d in all_dates
        if d < GDELT_EARLIEST and d not in already_done
    ]

    print(f"Dates to download: {len(dates_to_do)} | "
          f"Pre-GDELT zero-fill: {len(pre_gdelt)}")

    # Zero-fill pre-2013 dates
    if pre_gdelt:
        empty = pd.DataFrame(columns=['date', 'from_node', 'to_node',
                                       'GoldsteinScale', 'AvgTone', 'NumMentions'])
        pre_frames = [fill_all_pairs(empty, d) for d in pre_gdelt]
        all_results.append(pd.concat(pre_frames, ignore_index=True))
        print(f"  Zero-filled {len(pre_gdelt)} pre-GDELT dates")

    # Download and process each date
    for idx, date_str in enumerate(dates_to_do):
        day_df = download_and_parse_day(date_str)
        all_results.append(day_df)

        if (idx + 1) % 100 == 0:
            combined = pd.concat(all_results, ignore_index=True)
            combined = combined.drop_duplicates(
                subset=['date', 'from_node', 'to_node'], keep='last')
            combined.to_parquet(GDELT_TENSOR_FILE, index=False)
            n_signal = (combined['NumMentions'] > 0).sum()
            print(f"  Progress: {idx+1}/{len(dates_to_do)} | "
                  f"rows with signal: {n_signal}")

    # Final save
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined = combined.drop_duplicates(
            subset=['date', 'from_node', 'to_node'], keep='last')
        combined = combined.sort_values(
            ['date', 'from_node', 'to_node']).reset_index(drop=True)
        combined.to_parquet(GDELT_TENSOR_FILE, index=False)
        n_signal = (combined['NumMentions'] > 0).sum()
        print(f"\nDone! {len(combined)} rows, {n_signal} with real signal")
        print(f"Saved to {GDELT_TENSOR_FILE}")
        # Show a sample of non-zero rows
        sig = combined[combined['NumMentions'] > 0]
        if not sig.empty:
            print(sig.head(10).to_string())
    else:
        print("No data collected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end",   type=str, default=None)
    args = parser.parse_args()
    main(start_override=args.start, end_override=args.end)
