# GeoRipNet — Review Agent

## Role
Validate methodology, review code for correctness, and flag issues before they reach the paper.

## Review Checklist — Data

### Price Data
- [ ] WTI negative price on 2020-04-20 (-$36.98) handled correctly in model (normalisation/scaling)
- [ ] Urals interim spread schedule documented in paper with explicit caveat
- [ ] Indian Basket: verify monthly deviations from PPAC remain 0.0000 after any update
- [ ] `aligned_prices.parquet`: zero NaN in [2010-01-04, 2026-03-23] model window

### Comtrade Adjacency
- [ ] All adjacency matrices are row-normalised (rows sum to ≤1.0)
- [ ] RUS→IND weight jump verified: ~31% Jan 2019 → ~100% Jun 2022 (key paper signal)
- [ ] Pre-2018 back-fill documented in paper: "for training dates prior to Jan 2018, the Jan 2018 adjacency matrix is used as a static proxy"
- [ ] After 2010-2017 download: re-check gap count = 0

### GDELT
- [ ] CAMEO codes used: 13, 17, 18, 19, 20 only (conflict/hostile events — justified in paper)
- [ ] Actor pairs correctly mapped to node indices
- [ ] No lookahead: GDELT for day t uses events published on/before day t

## Review Checklist — Code

### Model
- [ ] GAT masking: isolated nodes (all-zero row in A_dynamic) → nan_to_num(0) → no NaN gradients
- [ ] TransformerEncoder: batch_first=True confirmed
- [ ] pos_enc shape [1, 200, d_model] — lookback window must be ≤ 200
- [ ] GeoRipNet.forward: a_dyn applied same for ALL k timesteps in lookback (correct — adjacency is daily, GDELT is for day t)

### Training
- [ ] Data leakage check: test set starts 2022-01-01, training ends 2019-12-31 — no overlap
- [ ] Scaler fitted only on training data, applied to val/test
- [ ] Gradient clipping at 1.0 — check for exploding gradients in first epoch

### Ablation Experiments
Required for paper (Table 3):
| Ablation | What changes |
|----------|-------------|
| full | No change |
| no_gdelt | Set gdelt=zeros → gate=0.5 → A_dynamic = 0.5 × A_static |
| no_dynamic_graph | Replace A_dynamic with identity matrix |
| gcn | Replace GAT with simple GCN |
| lstm | Replace Transformer with LSTM |
| single_benchmark | Predict only WTI (baseline) |

## Common Pitfalls to Catch
1. Using `df.columns = [...]` to rename Comtrade columns — use `.rename()` instead (column order varies)
2. ffill beyond 5 days on price data — check max gap
3. Comtrade period as float (pandas may read YYYYMM as float) — must cast to int
4. GDELT actor matching: NOR actors may be labelled NOR or GBR — verify mapping

## Review Triggers
Run this checklist whenever:
- Any data file is updated
- Model architecture is changed
- A new script is added to src/
- Before any paper figure is generated
