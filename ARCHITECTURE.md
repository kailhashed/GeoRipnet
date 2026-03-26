# Architecture Reference — GeoRipNet

Last updated: 2026-03-26
Source: `generation/architecture.md` (canonical spec)

---

## Task

Jointly predict next-day closing price for all 5 crude oil benchmarks:
```
Output: ŷ ∈ R^5  →  [WTI, Brent, OPEC, Urals, Indian Basket] for day t+1
```

Central hypothesis: geopolitical events affect prices **indirectly** by disrupting trade routes, not directly. The model encodes this mechanism explicitly.

---

## Node Map

| Node | Benchmark | Geopolitical Actor | Comtrade Country |
|------|-----------|--------------------|-----------------|
| 0 | WTI | United States | USA |
| 1 | Brent | North Sea | GBR + NOR |
| 2 | OPEC Basket | Saudi Arabia / Gulf | SAU |
| 3 | Urals | Russia | RUS |
| 4 | Indian Basket | India | IND |

---

## Inputs Per Training Sample

```
P(t-k : t)   ∈ R^{k × 5}       Historical daily prices, all 5 benchmarks
                                 k ∈ {10, 20, 30} (ablated)

E(t)         ∈ R^{5 × 5 × 3}   GDELT daily tensor for day t
                                 [i, j, c] = channel c for Actor1=node_i, Actor2=node_j
                                 Channels: GoldsteinScale | AvgTone | NumMentions

A_static     ∈ R^{5 × 5}        UN Comtrade HS-2709 monthly adjacency
                                 Row-normalised: A[i,j] = fraction of node_i exports to node_j
                                 Updated monthly via step function
```

---

## Module 1 — Geopolitical Edge Gating (Novel Contribution)

**What it does:** Produces a daily dynamic adjacency matrix from the geopolitical disruption signal.

```
W_g   ∈ R^{3→1}                Learned weight per GDELT channel
Gate(t) = σ(W_g · E(t) + b)    Shape [5×5], values ∈ (0,1)
A_dynamic(t) = A_static ⊙ Gate(t)
```

- Gate ≈ 0: geopolitics suppressing that trade edge (e.g., sanctions on Urals→Brent)
- Gate ≈ 1: edge operating at full baseline capacity
- **Key example**: Gate[3,4] (Urals→India) should show sustained elevation from Feb 2022 as sanctions rerouted Russian crude toward India

---

## Module 2 — Graph Attention Spatial Encoding

```
H(t) = GAT(P(t), A_dynamic(t))     Shape [5 × d_model]
```

- 2 attention heads
- Attention only along non-zero edges in A_dynamic(t)
- Run for all k days in window → sequence [H(t-k), ..., H(t)]

---

## Module 3 — Transformer Temporal Encoder

```
Input:  sequence [H(t-k), ..., H(t)]  flattened → [k × (5·d_model)]
Add positional encoding
Z = TransformerEncoder(sequence)       Shape [5·d_model]
```

Advantage over LSTM: self-attention can weight a structural break day (sanctions announcement) more than yesterday's marginal move.

---

## Module 4 — Multi-Output Prediction Head

```
LayerNorm(Z) → Dense(256) → ReLU → Dense(5)   [no final activation]
Output: ŷ ∈ R^5
```

---

## Training Configuration

| Split | Period | Purpose |
|-------|--------|---------|
| Train | 2010–2019 | Includes Crimea (2014), OPEC war (2016) |
| Validation | 2020–2021 | COVID crash, OPEC+ negotiations |
| Test | 2022–present | Russia-Ukraine sanctions — primary stress test |

**Loss**: MSE averaged across all 5 outputs: `L = (1/5) Σᵢ (ŷᵢ − yᵢ)²`

**Lookback ablation**: k ∈ {10, 20, 30} — trained and evaluated independently

---

## GDELT Tensor Construction

**Filters applied:**
- EventRootCode: 13 (diplomatic cooperation), 17 (sanctions), 18 (accusations), 19 (military threats), 20 (military conflict)
- Both Actor1 AND Actor2 must be in {USA, GBR, NOR, SAU, RUS, IND}

**Channels:**
| Channel | GDELT Field | Meaning |
|---------|-------------|---------|
| 0 | GoldsteinScale | Conflict intensity (-10 war → +10 cooperation) |
| 1 | AvgTone | Media sentiment (negative = hostile) |
| 2 | NumMentions | Global attention volume |

**Output**: Daily [N×76] parquet → reshaped to [5×5×3] per day during dataset build

---

## Ablation Experiments

| Experiment | What is removed | Metric |
|------------|----------------|--------|
| Full model | — | MAE, RMSE, MAPE per benchmark |
| No GDELT gating | Module 1 → A_static (fixed) | ΔRMSE |
| No dynamic graph | A_dynamic → fixed correlation matrix | ΔRMSE |
| GCN instead of GAT | Replace attention with standard graph conv | ΔRMSE |
| LSTM instead of Transformer | Replace Module 3 | ΔRMSE |
| Single benchmark | Separate model per benchmark, no joint prediction | ΔRMSE |

---

## Paper Figures

| Figure | What it shows |
|--------|--------------|
| 1 | Gate[3,4] (Urals→India) over 2021–2023: structural shift at Feb 2022 |
| 2 | GAT attention heatmap: 2022 crisis vs 2019 calm baseline |
| 3 | Predicted vs actual, all 5 benchmarks, 2022 test period |
| 4 | Ablation RMSE delta bar chart |

---

## File Structure (target)

```
georipnet/
├── data/
│   ├── price/                    # 5 benchmark CSVs
│   ├── gdelt_data/
│   │   ├── raw_cache/            # Cached daily/yearly parquet
│   │   └── daily_gdelt_tensor.parquet   # Final [N×76] output
│   └── uncomtrade/               # HS-2709 monthly adjacency matrices
├── src/
│   ├── collect_gdelt.py          # GDELT collector
│   ├── collect_comtrade.py       # Comtrade HS-2709 collector
│   ├── build_dataset.py          # Merge all → model-ready tensors
│   ├── model.py                  # Modules 1–4
│   ├── train.py                  # Training loop + checkpointing
│   └── evaluate.py               # Test eval + ablations + figures
├── generation/
│   └── architecture.md           # Canonical architecture spec (source of truth)
├── ARCHITECTURE.md               # This file (quick reference)
├── DATA_STATUS.md
├── DATA_SOURCES.md
├── METHODOLOGY.md
└── PROJECT_STATUS.md
```

---

## Novelty Claims Summary

1. **Mechanism-aware graph**: Comtrade bilateral flows as edges (not correlation matrices)
2. **Geopolitical edge gating**: GDELT modulates edge weights, not node features
3. **Multi-channel geopolitical tensor**: 3 orthogonal GDELT dimensions per country pair
4. **Joint 5-benchmark prediction**: Forces learning of inter-benchmark dependencies
5. **Mixed-frequency architecture**: Monthly trade structure + daily GDELT disruption, principled separation

**Target journal**: Energy Economics (Elsevier, IF ~13)
