# Data Sources — GeoRipNet

Last updated: 2026-03-26

---

## Price Data

### WTI (Node 0)
- **File**: `data/price/Cushing_OK_WTI_Spot_Price_FOB.csv`
- **Source**: U.S. Energy Information Administration (EIA)
- **URL**: https://www.eia.gov/dnav/pet/hist/RWTCD.htm
- **Series ID**: RWTCD (daily)
- **Units**: USD per barrel
- **Citation**: *U.S. Energy Information Administration, Cushing, OK WTI Spot Price FOB (Dollars per Barrel), retrieved [date].*

### Brent (Node 1)
- **File**: `data/price/Europe_Brent_Spot_Price_FOB.csv`
- **Source**: U.S. Energy Information Administration (EIA)
- **URL**: https://www.eia.gov/dnav/pet/hist/RBRTED.htm
- **Series ID**: RBRTED (daily)
- **Units**: USD per barrel
- **Citation**: *U.S. Energy Information Administration, Europe Brent Spot Price FOB (Dollars per Barrel), retrieved [date].*

### OPEC Basket (Node 2)
- **File**: `data/price/opec_basket.csv` (converted from `opec.xml`)
- **Source**: Organization of the Petroleum Exporting Countries (OPEC)
- **URL**: https://www.opec.org/opec_web/en/data_graphs/40.htm
- **Format**: Official XML (basketDayArchives.xsd schema)
- **Units**: USD per barrel
- **Citation**: *Organization of the Petroleum Exporting Countries, OPEC Reference Basket Price, retrieved [date].*

### Urals (Node 3)
- **File**: `data/price/urals_spot.csv` — PENDING
- **Source**: To be downloaded from Investing.com
- **Steps**: investing.com → search "Urals Oil" → Historical Data → download CSV
- **Expected range**: 2005-01-01 to present
- **Note**: `urals_oil_full.csv` is SYNTHETIC data — do not use
- **Citation (once sourced)**: *Investing.com, Urals Oil Historical Data, retrieved [date].*

### Indian Oil Basket (Node 4)
- **File**: `data/price/indian_basket_daily.csv` (reconstructed)
- **Primary source**: Petroleum Planning & Analysis Cell (PPAC), Government of India
- **PPAC file**: `data/price/1761715267_Crude-Oil-FOB-Price-(Indian Basket)-Historical.xlsx`
- **PPAC URL**: https://ppac.gov.in/prices/international-prices-of-crude-oil
- **Methodology**: Brent-scaled temporal disaggregation — see METHODOLOGY.md
- **Coverage**: 2000-04-03 to 2026-03-16
  - 2000-04 to 2025-03: Anchored to PPAC monthly averages (verified exact match)
  - 2025-04 to present: Extended using basket/brent ratio = 1.000080
- **Citation**: *Petroleum Planning and Analysis Cell (PPAC), Ministry of Petroleum and Natural Gas, Government of India, Crude Oil FOB Price (Indian Basket) — Historical, posted 01-04-2025. Daily values reconstructed via Brent-scaled temporal disaggregation following PPAC's published blend methodology.*

---

## Trade Flow Data

### UN Comtrade — HS Code 2709 (Crude Petroleum)
- **Files**: `data/uncomtrade/TradeData_*.csv`
- **Source**: UN Comtrade Database
- **HS Code**: 2709 (Petroleum oils and oils obtained from bituminous minerals, crude)
- **Countries**: USA, GBR, NOR, SAU, RUS, IND (bilateral flows between model nodes)
- **Granularity**: Monthly
- **Usage**: Static trade adjacency matrix A_static — updated monthly in model (step function)
- **Citation**: *United Nations Comtrade Database, Bilateral Trade in Crude Petroleum (HS 2709), retrieved [date].*

---

## Geopolitical Event Data

### GDELT 2.0
- **Status**: Not yet collected
- **Script**: `src/collect_gdelt.py`
- **Filters**:
  - EventRootCode: 13 (diplomatic cooperation), 17 (sanctions), 18 (accusations), 19 (military threats), 20 (military conflict)
  - Actor pairs: {USA, GBR, NOR, SAU, RUS, IND} × {USA, GBR, NOR, SAU, RUS, IND}
  - Channels: GoldsteinScale, AvgTone, NumMentions
- **Output shape**: Daily [5×5×3] tensor
- **Citation**: *Leetaru, K. & Schrodt, P.A. (2013). GDELT: Global Data on Events, Location, and Tone, 1979-2012. ISA Annual Convention.*

---

## EIA API

- **Key**: stored in `.env` as `EIA_API_KEY`
- **Note**: EIA API v2 only provides WTI and Brent spot prices. Dubai/Oman crude not available via API.
- **Available endpoints used**: `/v2/petroleum/pri/spt/data/`
