# Geopolitical Graph Neural Network for Crude Oil Benchmark Price Forecasting

## Overview

This paper presents a novel deep learning framework that models the full causal chain from geopolitical shock to trade route disruption to crude oil price movement across five major global benchmarks simultaneously. Unlike prior work that treats sentiment as a direct input feature, this architecture encodes geopolitical events as a dynamic disruption to the physical trade network — reflecting the actual economic mechanism by which sanctions, conflicts, and diplomatic events propagate into oil prices.

---

## Research Goal

To forecast the next-day closing price of all five major crude oil benchmarks jointly, using geopolitical event data and oil trade network structure as explicit graph-level signals rather than flat time-series features.

The central hypothesis is that geopolitical events do not affect oil prices directly — they affect prices *indirectly* by disrupting the trade routes through which oil physically flows between producer and consumer nations. Modelling this mechanism explicitly, rather than learning a statistical correlation, produces a more robust and interpretable forecasting system.

---

## Novelty Claims

### 1. Mechanism-Aware Graph Construction
Prior work either ignores trade network structure or uses static correlation matrices as graph edges. This architecture uses UN Comtrade bilateral crude oil trade flows (HS Code 2709) as the structural backbone of the graph — edges represent actual oil supply dependencies, not statistical correlations.

### 2. Geopolitical Edge Gating
The key novel contribution. Instead of appending GDELT sentiment as a node feature, geopolitical events are used to *modulate edge weights* in the trade graph daily. A learned gating function determines how much each GDELT channel (Goldstein Scale, AvgTone, NumMentions) suppresses or amplifies each bilateral trade edge on a given day. This directly models the mechanism: a sanctions event suppresses the Urals→Brent edge; a diplomatic agreement amplifies the SAU→IND edge.

### 3. Multi-Channel Geopolitical Tensor
GDELT data is encoded as a directed [5×5×3] tensor per day capturing three orthogonal dimensions of geopolitical activity per country pair: event severity (Goldstein), media framing (AvgTone), and global attention (NumMentions). Most prior work collapses these into a single sentiment score.

### 4. Simultaneous Five-Benchmark Prediction
No existing paper jointly predicts WTI, Brent, OPEC Basket, Urals, and Indian Oil Basket in a single model. Joint prediction forces the model to learn inter-benchmark dependencies, which is itself a finding — the learned attention weights reveal which benchmark pairs are most coupled under different geopolitical regimes.

### 5. Mixed-Frequency Architecture with Principled Separation
Monthly trade data and daily geopolitical data operate at different timescales. The architecture separates these explicitly: Comtrade provides the structural baseline (updated monthly via step function), while GDELT provides the daily disruption signal. This maps to economic reality and is clearly defensible to reviewers.

---

## Data Sources

| Data | Source | Granularity | Role |
|------|--------|-------------|------|
| Crude oil benchmark prices | Existing dataset | Daily | Price history input P(t−k:t) |
| Geopolitical event tensor | GDELT 2.0 (filtered) | Daily | Edge disruption signal E(t) |
| Oil trade flow adjacency | UN Comtrade HS-2709 | Monthly | Structural graph A_static |

### Price Nodes (Benchmarks)

| Node Index | Benchmark | Geopolitical Actor | Comtrade Country |
|------------|-----------|--------------------|-----------------|
| 0 | WTI | United States | USA |
| 1 | Brent | North Sea (UK + Norway) | GBR + NOR |
| 2 | OPEC Basket | Saudi Arabia / Gulf | SAU |
| 3 | Urals | Russia | RUS |
| 4 | Indian Oil Basket | India | IND |

### GDELT Tensor Channels

| Channel | GDELT Field | Description |
|---------|-------------|-------------|
| 0 | GoldsteinScale | Theoretical conflict intensity of event type. Range −10 (war) to +10 (full cooperation). Fixed per CAMEO event code. |
| 1 | AvgTone | Sentiment of news articles covering the event. Negative = hostile framing. |
| 2 | NumMentions | Total number of source mentions. Captures market attention volume. |

**GDELT CAMEO filters applied:**
- EventRootCode 13 — Diplomatic cooperation
- EventRootCode 17 — Sanctions and embargos
- EventRootCode 18 — Accusations and threats
- EventRootCode 19 — Military threats
- EventRootCode 20 — Military conflict

Only events where both Actor1 and Actor2 are among {USA, GBR, NOR, SAU, RUS, IND} are retained.

### Mixed-Frequency Handling

Comtrade data is monthly. The architecture uses a step function: the trade adjacency matrix for a given month is held constant for all days within that month and updated on the first day of the following month. Daily variation in the effective graph comes entirely from the GDELT gating signal. This is the principled choice because trade infrastructure changes slowly while geopolitical shocks are daily events.

---

## Architecture

### Inputs (per training sample)

```
P(t−k : t)   ∈ R^{k × 5}       Historical daily prices, all 5 benchmarks
                                 k ∈ {10, 20, 30} days (ablated)

E(t)         ∈ R^{5 × 5 × 3}   GDELT daily tensor for day t
                                 Directed: entry [i,j,c] = channel c value
                                 for events with Actor1=node_i, Actor2=node_j

A_static     ∈ R^{5 × 5}        UN Comtrade HS-2709 adjacency
                                 Row-normalised, updated monthly
                                 A_static[i,j] = fraction of node i's crude
                                 exports received by node j
```

---

### Module 1 — Geopolitical Edge Gating

**Purpose:** Produce a daily dynamic adjacency matrix that reflects the current geopolitical disruption to trade routes.

**Inputs:** E(t) ∈ R^{5×5×3},  A_static ∈ R^{5×5}

**Operation:**
```
W_g   ∈ R^{3 → 1}              Learnable weight per edge, collapses 3 channels to scalar
Gate(t) = σ( W_g · E(t) + b )  Shape [5 × 5], values in (0, 1)
A_dynamic(t) = A_static ⊙ Gate(t)
```

**Output:** A_dynamic(t) ∈ R^{5×5}

**Interpretation:** Gate(t)[i,j] close to 0 means the geopolitical climate on day t is suppressing the i→j trade edge (e.g. active sanctions). Gate(t)[i,j] close to 1 means the edge is operating at full baseline capacity. The element-wise product preserves the structural magnitude from Comtrade while allowing daily disruption from GDELT.

**Key example:** The Urals→India edge (node 3 → node 4) should show a persistent gate shift beginning February 2022 as EU sanctions redirected Russian crude toward India — visible as a sustained elevation in Gate(3,4) coinciding with suppression of Gate(3,1) (Urals→Brent).

---

### Module 2 — Graph Attention Spatial Encoding

**Purpose:** For each day t in the lookback window, compute a spatial embedding that captures how price signals propagate across the geopolitically-adjusted trade network.

**Inputs:** P(t) ∈ R^5,  A_dynamic(t) ∈ R^{5×5}

**Operation:**
```
H(t) = GAT( P(t), A_dynamic(t) )     Shape [5 × d_model]
```

Uses 2 attention heads. A_dynamic(t) serves as the adjacency mask — attention is only permitted along edges with non-zero weight in the dynamic graph. Each attention head learns a different weighting of the trade relationships under the current geopolitical conditions.

**Output:** H(t) ∈ R^{5 × d_model}  —  spatial snapshot of the global oil market for day t

**Run for all k days in the window** to produce sequence [H(t−k), H(t−k+1), ..., H(t)]

---

### Module 3 — Transformer Temporal Encoder

**Purpose:** Learn which historical market states in the lookback window are most predictive of tomorrow's prices, with special sensitivity to discontinuous geopolitical shock events.

**Input:** Sequence [H(t−k), ..., H(t)], each flattened to R^{5·d_model}
Full sequence shape: [k × (5 · d_model)]

**Operation:**
```
Add positional encoding to preserve time order
Z = TransformerEncoder( sequence )         Shape [5 · d_model]
```

Self-attention over the k timesteps learns that the day sanctions were announced (a structural break) may be more predictive than yesterday's marginal price move. This is the key advantage over LSTM which weights recent states disproportionately.

**Output:** Context vector Z ∈ R^{5·d_model}

---

### Module 4 — Multi-Output Prediction Head

**Purpose:** Map the latent context vector to next-day prices for all 5 benchmarks.

**Input:** Z ∈ R^{5·d_model}

**Operation:**
```
LayerNorm(Z)
→ Dense(256) → ReLU
→ Dense(5)   [no final activation — raw price regression]
```

**Output:** ŷ ∈ R^5  —  predicted prices [WTI, Brent, OPEC, Urals, Indian Basket] for day t+1

---

## Input → Output Summary

```
INPUT
─────
Historical prices    P(t−k:t)      [k × 5]           daily
GDELT tensor         E(t)          [5 × 5 × 3]        daily
Trade adjacency      A_static      [5 × 5]            monthly (step function)

PROCESSING
──────────
Module 1   Edge gating       A_dynamic(t)   [5 × 5]          daily
Module 2   GAT encoding      H(t)           [5 × d_model]    per day in window
Module 3   Temporal encode   Z              [5 · d_model]    per sample
Module 4   Prediction head   ŷ              [5]              next day

OUTPUT
──────
Predicted next-day closing price for:
  WTI  ·  Brent  ·  OPEC Basket  ·  Urals  ·  Indian Oil Basket
```

---

## Training Configuration

### Dataset

| Split | Period | Rationale |
|-------|--------|-----------|
| Train | 2010–2019 | Full pre-COVID baseline, includes Crimea (2014), OPEC war (2016) |
| Validation | 2020–2021 | COVID crash, OPEC+ negotiations |
| Test | 2022–present | Russia-Ukraine sanctions — the primary stress test |

Time order is strictly preserved. No shuffling across split boundaries.

### Loss Function

Mean Squared Error across all 5 outputs simultaneously:

```
L = (1/5) Σ_i ( ŷ_i − y_i )²
```

### Lookback Window Ablation

The model is trained and evaluated independently for k ∈ {10, 20, 30}. Results reported per benchmark and per k. The optimal k is expected to vary by benchmark — Urals likely requires longer memory post-2022 while WTI may respond within 10 days.

---

## Experimental Ablations

Each ablation removes one component to prove its individual contribution:

| Experiment | What is removed | Metric |
|------------|----------------|--------|
| Full model | — | MAE, RMSE, MAPE per benchmark |
| No GDELT gating | Module 1 replaced by A_static (no daily update) | Δ RMSE |
| No dynamic graph | A_dynamic replaced by fixed correlation matrix | Δ RMSE |
| GCN instead of GAT | Replace attention with standard graph convolution | Δ RMSE |
| LSTM instead of Transformer | Replace Module 3 | Δ RMSE |
| Single benchmark | Train separate model per benchmark, no joint prediction | Δ RMSE |

---

## Key Visualisations (Paper Figures)

**Figure 1 — Gate value on Urals→India edge (2021–2023)**
Plot Gate(t)[3,4] over time. Should show a structural shift beginning February 2022 and remaining elevated. This is the central qualitative finding — the model learned the sanctions-driven trade rerouting without being explicitly told.

**Figure 2 — Attention weights during crisis vs calm periods**
Heatmap of GAT attention weights during the 2022 sanctions period vs a calm 2019 baseline. Shows which benchmark pairs the model treats as coupled under geopolitical stress.

**Figure 3 — Prediction vs actual across all 5 benchmarks**
Time series of predicted vs actual prices for the 2022 test period, highlighting model accuracy around the February 2022 invasion event.

**Figure 4 — Ablation bar chart**
RMSE delta for each ablation experiment, demonstrating the contribution of each architectural component.

---

## Target Journals

| Journal | Publisher | Impact Factor | Primary Fit |
|---------|-----------|---------------|-------------|
| Energy Economics | Elsevier | ~13 | Geopolitics + quantitative energy modelling |
| Applied Energy | Elsevier | ~11 | Novel forecasting architecture |
| Expert Systems with Applications | Elsevier | ~8 | ML/graph learning contribution |
| Resources Policy | Elsevier | ~7 | Commodity markets + geopolitical risk |

**Primary submission target: Energy Economics**

---

## File Structure

```
project/
│
├── data/
│   ├── prices/                  # WTI, Brent, OPEC, Urals, Indian basket CSVs
│   ├── gdelt_data/
│   │   ├── meta/                # GDELT master file lists
│   │   ├── raw_cache/           # Cached raw daily/yearly parquet files
│   │   └── daily_gdelt_tensor.parquet   # Final [N × 76] output
│   └── comtrade/                # UN Comtrade HS-2709 monthly adjacency matrices
│
├── src/
│   ├── collect_gdelt.py         # GDELT data collector
│   ├── collect_comtrade.py      # Comtrade HS-2709 collector
│   ├── build_dataset.py         # Merge prices + GDELT + Comtrade → model-ready tensors
│   ├── model.py                 # Full architecture (Modules 1–4)
│   ├── train.py                 # Training loop, validation, checkpointing
│   └── evaluate.py              # Test set evaluation, ablations, figure generation
│
└── architecture.md              # This document
```