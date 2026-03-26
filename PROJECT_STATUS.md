# Project Status — GeoRipNet

Last updated: 2026-03-26

---

## What This Is

**GeoRipNet** — Geopolitical Graph Neural Network for Crude Oil Benchmark Price Forecasting

A deep learning model that predicts next-day closing prices for 5 crude oil benchmarks simultaneously, using geopolitical events (GDELT) to dynamically modulate bilateral oil trade edges (UN Comtrade), rather than treating sentiment as a flat input feature.

**Target journal**: Energy Economics (Elsevier, IF ~13)

Full architecture: see `generation/architecture.md`

---

## Node Map

| Node | Benchmark | Geopolitical Actor | Status |
|------|-----------|-------------------|--------|
| 0 | WTI | USA | Data ready |
| 1 | Brent | UK/Norway | Data ready |
| 2 | OPEC Basket | Saudi Arabia | Data ready |
| 3 | Urals | Russia | **BLOCKED — needs real data** |
| 4 | Indian Basket | India | Data ready (reconstructed) |

---

## Phase 1 — Data Collection

### Price Data
- [x] WTI daily (EIA CSV) — 1986–2026
- [x] Brent daily (EIA CSV) — 1987–2026
- [x] OPEC Basket daily (OPEC XML → CSV) — 2003–2026
- [ ] **Urals daily — PENDING** (download from Investing.com)
- [x] Indian Basket daily (reconstructed from PPAC + Brent) — 2000–2026

### Geopolitical Data
- [ ] GDELT tensor collection (`src/collect_gdelt.py`)

### Trade Flow Data
- [x] UN Comtrade HS-2709 CSVs downloaded
- [ ] Parse into monthly 5×5 adjacency matrices

---

## Phase 2 — Data Pipeline

- [ ] `src/build_dataset.py` — merge all sources into aligned tensors
- [ ] Normalisation / missing value handling
- [ ] Train/val/test split (2010–2019 / 2020–2021 / 2022–present)

---

## Phase 3 — Model

- [ ] `src/model.py` — implement Modules 1–4
- [ ] `src/train.py` — training loop
- [ ] `src/evaluate.py` — test evaluation + ablations

---

## Phase 4 — Results & Paper

- [ ] Figure 1: Gate value on Urals→India edge (2021–2023)
- [ ] Figure 2: GAT attention weights during crisis vs calm
- [ ] Figure 3: Prediction vs actual — all 5 benchmarks, 2022 test period
- [ ] Figure 4: Ablation bar chart

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| 5 nodes (not 6) | Daqing/China dropped — no free daily price, Dubai overlaps OPEC basket |
| Indian basket reconstructed | PPAC only publishes monthly; Brent-scaled disaggregation gives exact monthly match |
| PPAC formula used as-is | PPAC has not updated weights for Russian crude despite ~36% import share post-2022; divergence is noted as a limitation |
| Two-regime formula dropped | Would diverge from official PPAC benchmark; post-2022 discrepancy noted as limitation in paper |
| Urals synthetic data rejected | `urals_oil_full.csv` has 12-place decimals and wrong price levels throughout |
| OPEC XML → CSV converted | Browser-added header in XML prevented parsing; regex extraction used |

---

## Data Integrity Findings

### Verified OK
- WTI: Official EIA data, correct price levels
- Brent: Official EIA data, correct price levels
- OPEC Basket: Official OPEC XML, prices verified against Brent cross-check
- Indian Basket: Monthly averages match PPAC exactly (0.0000 deviation)

### Issues Found & Resolved
- `opec_2005_now.csv`: Replaced by `opec_basket.csv` (official source)
- `urals_oil_full.csv`: Confirmed synthetic — to be replaced

### Open Issues
- OPEC basket shows sharp spike Mar 2026 ($96 → $146) — authentic per OPEC source
- WTI/Brent end at 2026-03-16; OPEC/Indian basket extend to 2026-03-25/16
- Indian basket Apr 2025–present uses ratio-based extension (basket/brent = 1.000080)

---

## File Structure (current)

```
georipnet/
├── .env                          # EIA API key
├── DATA_STATUS.md                # Per-file status and coverage
├── DATA_SOURCES.md               # Citations and source URLs
├── METHODOLOGY.md                # Reconstruction methods and decisions
├── PROJECT_STATUS.md             # This file
├── data/
│   ├── price/
│   │   ├── Cushing_OK_WTI_Spot_Price_FOB.csv     # Node 0 - READY
│   │   ├── Europe_Brent_Spot_Price_FOB.csv        # Node 1 - READY
│   │   ├── opec_basket.csv                        # Node 2 - READY
│   │   ├── indian_basket_daily.csv                # Node 4 - READY
│   │   ├── urals_spot.csv                         # Node 3 - PENDING
│   │   ├── 1761715267_Crude-Oil-FOB-Price-...xlsx # PPAC source (keep)
│   │   ├── opec.xml                               # OPEC source (keep)
│   │   └── urals_oil_full.csv                     # SYNTHETIC - do not use
│   └── uncomtrade/
│       └── TradeData_*.csv                        # Trade flows - needs parsing
└── generation/
    └── architecture.md                            # Full model spec
```
