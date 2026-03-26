"""
collect_comtrade_2010_2017.py
Download UN Comtrade HS-2709 monthly data for 2010-2017.
Uses comtradeapicall.previewFinalData (no API key required, max 500 rows/call).

Reporters: USA=842, GBR=826, NOR=578, RUS=643, IND=356
Partners:  same 5 + SAU=682 (SAU never reports, captured via imports)

Output: data/uncomtrade/TradeData_2010_2017.csv
  — same shifted-column format as existing TradeData_*.csv files so
    build_adjacency.py works unchanged.
"""
import time
import pandas as pd
import comtradeapicall
from pathlib import Path

COMTRADE_DIR = Path(__file__).parent.parent / "data" / "uncomtrade"

# Reporter numeric codes
REPORTERS = {
    'USA': '842',
    'GBR': '826',
    'NOR': '578',
    'RUS': '643',
    'IND': '356',
}

# Partner codes to request (our nodes + SAU)
PARTNER_CODES = '842,826,578,643,356,682'

YEARS  = range(2010, 2018)   # 2010..2017 inclusive
MONTHS = range(1, 13)


def fetch_one(reporter_code: str, period: str) -> pd.DataFrame | None:
    """Fetch HS-2709 flows for one reporter × one month. Returns None on failure."""
    try:
        df = comtradeapicall.previewFinalData(
            typeCode='C', freqCode='M', clCode='HS',
            period=period,
            reporterCode=reporter_code,
            cmdCode='2709',
            flowCode='X,M',
            partnerCode=PARTNER_CODES,
            partner2Code=None, customsCode=None, motCode=None,
            maxRecords=500, format_output='JSON',
            aggregateBy=None, breakdownMode='classic',
            countOnly=None, includeDesc=True,
        )
        return df
    except Exception as exc:
        print(f"  ERROR {reporter_code} {period}: {exc}")
        return None


def api_to_shifted(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert API response columns to the shifted format used by build_adjacency.py:
      reporterCode = ISO3 code   (API: reporterISO)
      partnerCode  = ISO3 code   (API: partnerISO)
      reporterDesc = flow code   (API: flowCode  — 'X' or 'M')
      refMonth     = YYYYMM int  (API: period)
      fobvalue     = FOB USD     (API: fobvalue)
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = pd.DataFrame({
        'reporterCode': df['reporterISO'],
        'partnerCode':  df['partnerISO'],
        'reporterDesc': df['flowCode'],
        'refMonth':     df['period'],
        'fobvalue':     pd.to_numeric(df['fobvalue'], errors='coerce').fillna(0),
    })
    return out[out['fobvalue'] > 0]   # drop zero-value rows


def main():
    all_rows = []
    total_calls = len(REPORTERS) * len(list(YEARS)) * 12
    done = 0

    for iso, code in REPORTERS.items():
        for year in YEARS:
            for month in MONTHS:
                period = f"{year}{month:02d}"
                df_raw = fetch_one(code, period)
                shifted = api_to_shifted(df_raw)
                if not shifted.empty:
                    all_rows.append(shifted)
                done += 1
                if done % 20 == 0:
                    print(f"  Progress: {done}/{total_calls}  last={iso} {period}  rows so far={sum(len(r) for r in all_rows)}")
                time.sleep(0.3)   # stay well under rate limit

    if not all_rows:
        print("No data collected — check API availability.")
        return

    combined = pd.concat(all_rows, ignore_index=True)
    out_path = COMTRADE_DIR / "TradeData_2010_2017.csv"
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined):,} rows -> {out_path}")

    # Quick sanity check
    combined['period_int'] = pd.to_numeric(combined['refMonth'], errors='coerce').astype(int)
    periods = sorted(combined['period_int'].unique())
    print(f"Period range: {periods[0]} to {periods[-1]}  ({len(periods)} months)")
    reporters = combined['reporterCode'].unique().tolist()
    print(f"Reporters in data: {reporters}")


if __name__ == "__main__":
    main()
