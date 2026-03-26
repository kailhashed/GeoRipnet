# Data Status — GeoRipNet

Last updated: 2026-03-26

---

## Price Data — 5 Benchmarks (all YYYY-MM-DD ascending, in data/price/)

| Node | Benchmark | File | Range | Status |
|------|-----------|------|-------|--------|
| 0 | WTI | `wti_daily.csv` | 1986-01-02 to 2026-03-16 | READY |
| 1 | Brent | `brent_daily.csv` | 1987-05-20 to 2026-03-16 | READY |
| 2 | OPEC Basket | `opec_daily.csv` | 2003-01-02 to 2026-03-25 | READY |
| 3 | Urals | `urals_daily.csv` | 2005-01-04 to 2026-03-16 | INTERIM (Brent-spread) |
| 4 | Indian Basket | `indian_basket_daily.csv` | 2000-04-03 to 2026-03-16 | READY (PPAC-anchored) |

**Unified aligned file**: `data/aligned_prices.parquet`
- 4,185 trading days in model window (2010-01-04 to 2026-03-23)
- Zero missing values across all 5 columns in model window
- Note: WTI hit -$36.98 on 2020-04-20 (real negative price event) — model must handle this

**Urals action required**: Replace `urals_daily.csv` with real data from Investing.com before paper submission.

---

## Comtrade Trade Adjacency

| File | Description | Status |
|------|-------------|--------|
| `data/uncomtrade/adjacency_monthly.parquet` | 5x5 normalised monthly matrices | READY |
| `data/uncomtrade/adjacency_monthly_readable.csv` | Human-readable version | READY |

- Coverage: **201801 to 202412** (84 months, Jan 2018 to Dec 2024, no gaps)
- Key signal verified: RUS→IND weight jumps from 31% (Jan 2019) to 100% (Jun 2022)
- **Gap: 2010–2017 not in Comtrade download** — model will back-fill with 201801 matrix for pre-2018 dates

**Fix needed**: Download Comtrade data for 2010–2017 to cover the full training window.
Steps: UN Comtrade → HS 2709 → reporters {USA,GBR,NOR,RUS,IND} → 2010-2017 → add to data/uncomtrade/

---

## GDELT Data

| File | Status |
|------|--------|
| `data/gdelt_data/daily_gdelt_tensor.parquet` | NOT YET COLLECTED |

**Next step**: Run `src/collect_gdelt.py` (to be written).

---

## Source Files (data/raw_sources/ — do not use directly)

| File | What it is |
|------|-----------|
| `Cushing_OK_WTI_Spot_Price_FOB.csv` | Original WTI download (MM/DD/YYYY desc) |
| `Europe_Brent_Spot_Price_FOB.csv` | Original Brent download (MM/DD/YYYY desc) |
| `opec.xml` | Original OPEC basket XML from opec.org |
| `opec_basket.csv` | Converted from opec.xml |
| `urals_interim.csv` | Brent-spread Urals construction |
| `1761715267_*.xlsx` | PPAC historical monthly (2000–2025) |
| `1774419685_*.xlsx` | PPAC current FY (2025-26, posted 25.03.2026) |

**Do not use**: `data/raw_backup/urals_oil_full.csv` — confirmed synthetic data

---

## Next Actions (Priority Order)

1. [ ] **GDELT collection** — write and run `src/collect_gdelt.py`
2. [ ] **Comtrade 2010–2017** — download pre-2018 data from UN Comtrade, re-run `build_adjacency.py`
3. [ ] **Replace Urals** — manual download from Investing.com → `data/price/urals_daily.csv`
4. [ ] **Build full dataset** — `python src/build_dataset.py` (after GDELT ready)
5. [ ] **Train model** — `python src/train.py --lookback 20`
