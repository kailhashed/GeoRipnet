"""
normalize_urals.py
Converts an Investing.com manual CSV download into the standard Date,Price format.

Investing.com exports look like:
  "Date","Price","Open","High","Low","Change %"
  "Mar 27, 2026","82.50","81.20","83.10","80.90","+1.60%"

Usage:
  python src/normalize_urals.py path/to/downloaded_file.csv
  python src/normalize_urals.py  (auto-detects raw file in data/price/)
"""
import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import PRICE_DIR

OUTPUT_FILE = PRICE_DIR / "urals_daily.csv"


def normalize(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    df.columns = [c.strip().strip('"') for c in df.columns]

    # Find date and price columns regardless of exact capitalisation
    date_col  = next(c for c in df.columns if c.lower() == "date")
    price_col = next(c for c in df.columns if c.lower() == "price")

    df["Date"]  = pd.to_datetime(df[date_col].str.strip().str.strip('"'),
                                  infer_datetime_format=True, errors="coerce")
    df["Price"] = (
        df[price_col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip().str.strip('"')
        .pipe(pd.to_numeric, errors="coerce")
    )

    df = df.dropna(subset=["Date", "Price"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    return df[["Date", "Price"]]


def main():
    if len(sys.argv) > 1:
        input_path = Path(sys.argv[1])
    else:
        # Auto-detect: look for any CSV in data/price/ that isn't already a clean file
        candidates = [
            p for p in PRICE_DIR.glob("*.csv")
            if p.name not in {
                "wti_daily.csv", "brent_daily.csv", "opec_daily.csv",
                "urals_daily.csv", "indian_basket_daily.csv"
            }
        ]
        if not candidates:
            print("Usage: python src/normalize_urals.py path/to/investing_download.csv")
            sys.exit(1)
        input_path = candidates[0]
        print(f"Auto-detected input: {input_path}")

    print(f"Reading: {input_path}")
    df = normalize(input_path)
    print(f"Parsed {len(df)} rows | {df['Date'].iloc[0]} → {df['Date'].iloc[-1]}")
    print(f"Price range: ${df['Price'].min():.2f} – ${df['Price'].max():.2f}")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved → {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
