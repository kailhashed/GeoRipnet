# GeoRipNet — Data Agent

## Role
Collect, validate, and maintain all datasets. Raise blockers to the Orchestrator.

## Datasets Under Management

### 1. Price Data (`data/price/`)
| File | Status | Action |
|------|--------|--------|
| `wti_daily.csv` | READY | None |
| `brent_daily.csv` | READY | None |
| `opec_daily.csv` | READY | None |
| `urals_daily.csv` | INTERIM (Brent-spread) | Replace with Investing.com download before submission |
| `indian_basket_daily.csv` | READY (PPAC-anchored) | None |
| `aligned_prices.parquet` | READY | 4,185 days, zero NaN in model window |

**Urals spread schedule used for interim**:
- Pre-2022: Brent − $1.50
- Apr–Jun 2022: Brent − $30.00
- Jul 2022–Dec 2023: Brent − $20.00
- Jan 2024–Dec 2024: Brent − $13.00
- Jan 2025–present: Brent − $2.00

### 2. Comtrade Adjacency (`data/uncomtrade/`)
| File | Status |
|------|--------|
| `adjacency_monthly.parquet` | READY for 2018-2024 (84 months) |
| `TradeData_2010_2017.csv` | IN PROGRESS (collecting) |

**Column format of TradeData_*.csv (SHIFTED — do not change)**:
- `reporterCode` = ISO3 of reporter (e.g. 'USA')
- `partnerCode`  = ISO3 of partner
- `reporterDesc` = flow code ('X' or 'M')
- `refMonth`     = period YYYYMM
- `fobvalue`     = FOB value USD

**Node map**: USA=0, GBR=1, NOR=1, SAU=2, RUS=3, IND=4
SAU never reports — captured via import records (reporter imports FROM SAU → swap direction).

After 2010-2017 collection: re-run `python src/build_adjacency.py`

### 3. GDELT Tensor (`data/gdelt_data/`)
| File | Status |
|------|--------|
| `daily_gdelt_tensor.parquet` | NOT COLLECTED |

**Next step**: Write and run `src/collect_gdelt.py`

**Tensor spec**: [date, from_node, to_node, channel]
- Shape per day: 5×5×3
- Channels: GoldsteinScale, AvgTone, NumMentions
- CAMEO codes: 13, 17, 18, 19, 20 (hostile/conflict events)
- Actor pairs from {USA, GBR, NOR, SAU, RUS, IND}
- Source: GDELT 2.0 GKG or events table (BigQuery or direct download)

## Validation Rules
1. All price files: ascending date, no duplicate dates, no NaN in model window
2. Adjacency: row-normalised (rows sum to 1 or 0), periods contiguous
3. GDELT: one row per (date, from_node, to_node), no gaps in trading days

## Scripts
- `src/build_prices.py` — Merge and align price CSVs → `aligned_prices.parquet`
- `src/build_adjacency.py` — Build adjacency matrices from TradeData_*.csv
- `src/collect_comtrade_2010_2017.py` — Download 2010-2017 Comtrade data
- `src/collect_gdelt.py` — (TO BE WRITTEN) Download GDELT tensor
- `src/build_dataset.py` — Merge all data → `dataset_k{lookback}.parquet`
