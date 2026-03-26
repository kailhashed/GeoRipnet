# GeoRipNet — Docs Agent

## Role
Maintain all documentation, draft paper sections, and ensure ARCHITECTURE.md / DATA_STATUS.md / PROJECT_STATUS.md stay current.

## Document Ownership

| Document | Purpose | Update trigger |
|----------|---------|----------------|
| `ARCHITECTURE.md` | Single source of truth for model | Any architectural change |
| `DATA_STATUS.md` | Per-file data status + next actions | Any data update |
| `DATA_SOURCES.md` | Citations and URLs for all datasets | New data source added |
| `METHODOLOGY.md` | Formulae, reconstruction logic | Methodology decision |
| `PROJECT_STATUS.md` | Phase tracker, decisions log | Phase change or major decision |
| `agents/*.md` | Agent instructions | Process change |

## Paper Outline (Energy Economics format)

### Abstract (~250 words)
- Problem: crude oil price forecasting across multiple benchmarks
- Gap: existing models ignore geopolitical event flow gating on trade networks
- Method: GeoRipNet — GAT + Transformer + GDELT gating on UN Comtrade adjacency
- Result: MAE/RMSE vs baselines on 2022-present test period

### 1. Introduction
- Multi-benchmark forecasting motivation
- Geopolitical disruption as structural break (Russia-Ukraine Feb 2022)
- Contribution list: (1) dynamic adjacency gating, (2) multi-benchmark joint prediction, (3) real post-invasion test period

### 2. Related Work
- Oil price forecasting: ML methods (LSTM, GRU, Transformer)
- Graph neural networks for financial/energy forecasting
- Geopolitical event incorporation in forecasting

### 3. Data
- 3.1 Price benchmarks (5 nodes, sources, coverage)
- 3.2 UN Comtrade bilateral trade adjacency (HS-2709, monthly, normalisation)
- 3.3 GDELT geopolitical event tensor (CAMEO codes, 3 channels, actor mapping)

### 4. Methodology
- 4.1 Problem formulation
- 4.2 Geopolitical Edge Gating (Module 1)
- 4.3 GAT Spatial Encoding (Module 2)
- 4.4 Transformer Temporal Encoder (Module 3)
- 4.5 Multi-output Prediction Head (Module 4)

### 5. Experiments
- 5.1 Baselines: LSTM, GRU, Transformer, GCN+LSTM
- 5.2 Main results (Table 1: MAE/RMSE/MAPE per benchmark)
- 5.3 Ablation study (Table 2)
- 5.4 Case study: Gate[Urals→India] edge over Russia-Ukraine period (Figure 1)

### 6. Conclusion

## Key Claims to Defend in Paper
1. "Indian basket is reconstructed via PPAC monthly anchor with Brent-scaled temporal disaggregation" — cite PPAC methodology document, verify 0.0000 monthly deviation
2. "Urals prices use Brent minus market spread" — cite spread schedule; replace with real data before submission
3. "Pre-2018 trade adjacency uses Jan-2018 matrix as static proxy" — explicitly state limitation
4. "GDELT CAMEO codes 13, 17, 18, 19, 20 selected to capture hostile events" — justify selection

## Style Rules
- Energy Economics: Elsevier, no word limit for research articles
- Equation numbering: (1), (2), ...
- Tables: MAE/RMSE/MAPE format, bold best result
- Figures: 300 DPI minimum; save in `results/figures/`

## Pending Documentation Tasks
- [ ] Update DATA_STATUS.md after 2010-2017 Comtrade download completes
- [ ] Update PROJECT_STATUS.md phase tracker after GDELT collection
- [ ] Draft Section 3.2 (Comtrade data) in paper draft
- [ ] Draft Section 4.2 (Edge Gating) math — matches model.py exactly
