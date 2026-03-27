"""
collect_urals.py
Scrapes daily Urals crude oil spot prices from Investing.com.

Uses the internal HistoricalDataAjax endpoint (same as the site's Download button).
Fetches in two chunks to stay under the 5,000-row page limit.

Output: data/price/urals_daily.csv
  Columns: Date (YYYY-MM-DD), Price (USD/bbl)
  Coverage: 2005-01-03 to present

Usage:
  python src/collect_urals.py
  python src/collect_urals.py --start 2005-01-01 --end 2026-12-31
"""
import sys
import time
import argparse
from datetime import datetime, date
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent))
from config import PRICE_DIR

AJAX_URL    = "https://www.investing.com/instruments/HistoricalDataAjax"
PAIR_ID     = "8849"
SML_ID      = "28717652"
OUTPUT_FILE = PRICE_DIR / "urals_daily.csv"

HEADERS = {
    "User-Agent":       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36",
    "X-Requested-With": "XMLHttpRequest",
    "Referer":          "https://www.investing.com/commodities/"
                        "crude-oil-urals-spot-futures-historical-data",
    "Content-Type":     "application/x-www-form-urlencoded",
    "Accept-Language":  "en-US,en;q=0.9",
}


def _get_session():
    try:
        import cloudscraper
        session = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        print("Using cloudscraper (Cloudflare bypass)")
        return session
    except ImportError:
        import requests
        session = requests.Session()
        print("cloudscraper not found, using requests (may be blocked by Cloudflare).")
        print("Install with:  pip install cloudscraper")
        return session


def _fetch_chunk(session, st_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch one chunk via the AJAX endpoint.
    Dates must be MM/DD/YYYY strings.
    Returns DataFrame with columns [Date, Price].
    """
    payload = {
        "curr_id":      PAIR_ID,
        "smlID":        SML_ID,
        "header":       "Crude Oil Urals Historical Data",
        "st_date":      st_date,
        "end_date":     end_date,
        "interval_sec": "Daily",
        "sort_col":     "date",
        "sort_ord":     "ASC",
        "action":       "historical_data",
    }

    resp = session.post(AJAX_URL, headers=HEADERS, data=payload, timeout=60)
    resp.raise_for_status()

    soup  = BeautifulSoup(resp.text, "lxml")
    table = soup.find("table", {"id": "curr_table"})
    if table is None:
        raise RuntimeError("No table found in response — Investing.com may have blocked the request.")

    rows = []
    for tr in table.find("tbody").find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all("td")]
        if len(cols) < 2:
            continue
        rows.append({"Date": cols[0], "Price": cols[1]})

    if not rows:
        return pd.DataFrame(columns=["Date", "Price"])

    df = pd.DataFrame(rows)
    df["Date"]  = pd.to_datetime(df["Date"], format="%b %d, %Y", errors="coerce")
    df["Price"] = (
        df["Price"]
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )
    df = df.dropna(subset=["Date", "Price"])
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df[["Date", "Price"]].copy()


def scrape(start: str = "2005-01-01", end: str = None) -> pd.DataFrame:
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    session = _get_session()

    def to_inv(s: str) -> str:
        """YYYY-MM-DD to MM/DD/YYYY"""
        return datetime.strptime(s, "%Y-%m-%d").strftime("%m/%d/%Y")

    # Split into two chunks to stay under the 5,000-row limit
    # (~3,100 trading days per chunk at 260/year)
    split = "2015-12-31"
    chunks = []
    if start <= split:
        chunk_end = split if end > split else end
        print(f"Chunk 1: {start} to {chunk_end} ...", end=" ", flush=True)
        df1 = _fetch_chunk(session, to_inv(start), to_inv(chunk_end))
        print(f"{len(df1)} rows")
        chunks.append(df1)
        time.sleep(2)

    if end > split:
        chunk_start = split if start <= split else start
        # Move chunk_start one day forward to avoid overlap
        chunk_start = (datetime.strptime(chunk_start, "%Y-%m-%d").strftime("%Y-%m-%d")
                       if start > split else
                       datetime.strptime(split, "%Y-%m-%d").strftime("%Y-%m-%d"))
        print(f"Chunk 2: {chunk_start} to {end} ...", end=" ", flush=True)
        df2 = _fetch_chunk(session, to_inv(chunk_start), to_inv(end))
        print(f"{len(df2)} rows")
        chunks.append(df2)

    if not chunks:
        return pd.DataFrame(columns=["Date", "Price"])

    df = pd.concat(chunks, ignore_index=True)
    df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def save(df: pd.DataFrame):
    PRICE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
    print(f"Date range : {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")


def _sanity_check(df: pd.DataFrame):
    checks = [
        ("2022-02-23", 85, 105, "day before invasion — should be near Brent ~99"),
        ("2022-02-24", 85, 115, "invasion day — Brent spiked to 101, Urals should follow"),
        ("2022-04-15", 55,  95, "peak sanctions — real discount was $25-35 below Brent"),
        ("2020-04-20", -5,  30, "WTI negative day — Urals distressed but not negative"),
    ]
    print("\nSanity checks (real data should show volatile spread post-invasion):")
    for d, lo, hi, note in checks:
        row = df[df["Date"] == d]
        if row.empty:
            print(f"  {d}: not in data")
            continue
        price = row["Price"].iloc[0]
        status = "OK" if lo <= price <= hi else "WARN"
        print(f"  {d}: ${price:.2f}  [{status}]  {note}")

    # Check that the spread is NOT a fixed step function post-invasion
    post = df[df["Date"] >= "2022-02-24"].head(30)
    if len(post) > 5:
        spread_variance = post["Price"].std()
        if spread_variance > 3.0:
            print(f"\n  Post-invasion price std=${spread_variance:.2f} — looks like real volatile data (good)")
        else:
            print(f"\n  WARNING: Post-invasion price std=${spread_variance:.2f} — suspiciously low, may be synthetic")


def main():
    parser = argparse.ArgumentParser(description="Scrape Urals crude oil prices from Investing.com")
    parser.add_argument("--start", default="2005-01-01")
    parser.add_argument("--end",   default=None)
    args = parser.parse_args()

    print(f"Collecting Urals daily prices: {args.start} to {args.end or 'today'}")
    df = scrape(start=args.start, end=args.end)

    if df.empty:
        print("No data collected.")
        sys.exit(1)

    save(df)
    _sanity_check(df)


if __name__ == "__main__":
    main()
