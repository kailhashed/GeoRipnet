# GeoRipNet — Code Agent

## Role
Implement, maintain, and debug all model code. Follow the architecture spec exactly.

## Architecture Reference: `ARCHITECTURE.md`
Do not modify the architecture without Orchestrator approval.

## Module Ownership

### `src/model.py` — GeoRipNet model
Four modules, in order:
1. `GeopoliticalEdgeGating` — GDELT [B,5,5,3] → gate [B,5,5] → A_dynamic = A_static ⊙ Gate
2. `GraphAttentionLayer` — multi-head GAT, attention masked to non-zero edges in A_dynamic
3. `TransformerTemporalEncoder` — learnable positional encoding + TransformerEncoder, returns last timestep
4. `GeoRipNet` — forward(prices[B,k,5], gdelt[B,5,5,3], a_static[B,5,5]) → [B,5]

**Critical invariants** (do not break):
- D_MODEL = 64 (set in config.py)
- N_HEADS_GAT = 4
- N_TRANSFORMER_LAYERS = 2
- N_NODES = 5
- Output: [B, 5] next-day prices for all 5 benchmarks

### `src/train.py` — Training loop
- Adam optimizer, ReduceLROnPlateau(patience=5, factor=0.5)
- Gradient clipping: max_norm=1.0
- MSELoss
- Early stopping: PATIENCE = 20
- Saves best checkpoint: `checkpoints/georipnet_k{lookback}_best.pt`

### `src/evaluate.py` — Evaluation
- Metrics: MAE, RMSE, MAPE per node + mean
- Ablations: full, no_gdelt, no_dynamic_graph, gcn, lstm, single_benchmark
- Figures: fig1_gate_urals_india.png, fig3_pred_vs_actual.png

### `src/build_dataset.py` — Dataset assembly
- Loads aligned_prices.parquet + daily_gdelt_tensor.parquet + adjacency_monthly.parquet
- Creates sliding window samples of size `lookback`
- Back-fills adjacency with Jan-2018 matrix for pre-2018 dates (until 2010-2017 data available)
- Saves `data/dataset_k{lookback}.parquet`

### `src/config.py` — Central config (do not scatter constants)

## Coding Rules
1. Never hard-code paths — use config.py constants
2. Never use N_NODES = 5 inline — import from config
3. All scripts must run from `src/` directory with `python script.py`
4. No notebooks in src/ — notebooks only in project root for exploration
5. GPU-compatible: use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

## Pending Implementation
- [ ] `src/collect_gdelt.py` — GDELT tensor collection (NEXT PRIORITY after Comtrade fix)
- [ ] `src/build_dataset.py` — needs GDELT to be ready first
