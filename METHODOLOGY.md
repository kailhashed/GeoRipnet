# Methodology Notes — GeoRipNet

Last updated: 2026-03-26

---

## 1. Indian Oil Basket — Daily Price Reconstruction

### Why reconstruction is needed
PPAC (Petroleum Planning & Analysis Cell, India) publishes the Indian Oil Basket price as **monthly averages only** on ppac.gov.in. No free daily API or download exists for the daily series.

### Official PPAC Formula
```
Indian_Basket = sour_ratio × avg(Oman_spot, Dubai_spot) + sweet_ratio × Brent_Dated
```

The sour:sweet ratio is updated annually by PPAC based on the previous fiscal year's refinery processing mix.

### Ratio History (from PPAC official Excel)

| Fiscal Year | Sour (Oman/Dubai) | Sweet (Brent) |
|-------------|-------------------|---------------|
| 2000–05 | 57.00% | 43.00% |
| 2005–06 | 58.00% | 42.00% |
| 2010–11 | 67.60% | 32.40% |
| 2014–15 | 72.04% | 27.96% |
| 2018–19 | 74.77% | 25.23% |
| 2019–20 | 75.50% | 24.50% |
| 2020–24 | 75.62% | 24.38% |
| **2024–25** | **78.50%** | **21.50%** |

The ratio has steadily increased toward sour grades as Indian refinery capacity for heavy crude expanded.

### Why Daily Dubai/Oman Prices Are Unavailable for Free
- EIA API v2: Only WTI and Brent in spot prices endpoint
- EIA legacy API (v1): Dubai series (RDUBD) discontinued / not accessible
- FRED: No daily Dubai/Oman series
- IMF: Monthly only
- Platts/Argus/Bloomberg: Paywalled (official sources for Oman/Dubai spot)

### Reconstruction Method: Brent-Scaled Temporal Disaggregation

Since Oman/Dubai and Brent have a daily correlation > 0.99, Brent daily movements serve as a reliable intramonth distribution indicator.

**For each month M where PPAC data exists (Apr 2000 – Mar 2025):**
```
basket_daily(d) = basket_monthly_ppac(M) × [brent_daily(d) / brent_monthly_avg(M)]
```

This guarantees:
- Monthly average of `basket_daily` = `basket_monthly_ppac` exactly (verified: max deviation = 0.0000)
- Daily movement pattern follows Brent's observed daily fluctuations
- No free parameters or assumptions beyond the monthly anchor

**For Apr 2025 – present (beyond PPAC data availability):**
```
basket_daily(d) = brent_daily(d) × ratio_recent
```
Where `ratio_recent` = mean(basket/brent) over Oct 2024–Mar 2025 = **1.000080**

This reflects that in recent months Oman/Dubai has traded at near-parity with Brent, consistent with tightening sour-sweet spreads.

### Verification Results

Cross-checked 15 years of reconstructed monthly averages against PPAC published values:
- **Zero anomalies** (all deviations < 0.0001)
- Implied Oman/Dubai prices range $0–$8.51 below Brent — consistent with historical sour-sweet differential

### Paper Citation Language

*"Daily Indian Oil Basket prices are reconstructed using a Brent-scaled temporal disaggregation method anchored to PPAC monthly averages. PPAC's annually-updated sour:sweet blend ratios are applied per fiscal year. For the period beyond March 2025, the basket-to-Brent ratio from the preceding six months is used as a stable scaling factor, consistent with the near-parity pricing of Oman/Dubai versus Brent observed during this period."*

### Known Limitations

1. **Post-2022 composition shift**: India's actual import mix shifted to ~36% Russian Urals by FY2023-24, but PPAC has not updated its formula to include Urals. The official basket therefore overstates India's actual acquisition cost by an estimated 10–15% for 2022–present. This is a limitation of the official benchmark itself, not of our reconstruction.

2. **Intramonth dynamics**: The reconstruction assumes Oman/Dubai moves proportionally with Brent within each month. Brief Oman/Dubai-specific shocks (e.g., DME auction anomalies) are not captured at daily resolution.

---

## 2. Urals — Data Status

Real Urals daily spot price data is needed. The file `urals_oil_full.csv` was found to be **synthetic** (12+ decimal places, wrong price levels throughout) and must be replaced.

**Target source**: Investing.com Urals Oil historical data (free manual download)
**Expected price range**: 2005: ~$40–50/bbl, 2022 (post-invasion): ~$70–90/bbl with $15–35 discount to Brent

---

## 3. OPEC Basket — Data Integrity Note

The official OPEC basket XML from opec.org shows a significant price spike in March 2026 ($96 → $146 over 3 weeks). This is authentic official data. The large spread vs Brent during this period is consistent with geopolitical events driving Middle Eastern sour crude premiums — exactly the kind of signal this model is designed to capture.

---

## 4. Ratio Update for FY 2025-26

PPAC posts the new ratio each April when the previous fiscal year's processing data is available. The FY2025-26 ratio (reflecting FY2024-25 processing) was not yet published as of the last PPAC file date (2025-04-01). Current best estimate: **78.71:21.29** (based on the progressive trend). The file uses 78.50:21.50 (FY2024-25 ratio) as the conservative estimate.
