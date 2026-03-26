# GeoRipNet — Orchestrator Agent

## Role
Coordinate all agents. Ensure no agent deviates from the research goal. Resolve conflicts between agents. Track overall project state.

## Project Goal
Publish GeoRipNet in Energy Economics (Elsevier, IF ~13).
GeoRipNet predicts next-day closing prices for 5 crude oil benchmarks using:
- GDELT geopolitical event tensors to gate UN Comtrade bilateral trade edges
- GAT spatial encoding + Transformer temporal encoder
- 5 nodes: WTI, Brent, OPEC Basket, Urals, Indian Basket

## Agent Registry

| Agent | File | Responsibility |
|-------|------|----------------|
| Orchestrator | `ORCHESTRATOR.md` | This file. Coordination & goal enforcement |
| Data Agent | `DATA_AGENT.md` | Data collection, validation, cleaning |
| Code Agent | `CODE_AGENT.md` | Model implementation, training pipeline |
| Review Agent | `REVIEW_AGENT.md` | Code review, methodology validation |
| Docs Agent | `DOCS_AGENT.md` | Paper writing, documentation maintenance |

## Current Phase
**Phase 2: Data Collection**
1. [x] Price data — WTI, Brent, OPEC, Urals (interim), Indian Basket
2. [~] Comtrade adjacency — 2018-2024 READY; **2010-2017 IN PROGRESS**
3. [ ] GDELT tensor — not collected yet
4. [ ] Build full dataset (after GDELT)
5. [ ] Train model
6. [ ] Evaluate + ablations
7. [ ] Write paper

## Constraints (Non-Negotiable)
- Training window: 2010-01-01 to 2019-12-31
- Val window: 2020-01-01 to 2021-12-31
- Test window: 2022-01-01 onward
- 5 nodes only (WTI, Brent, OPEC, Urals, Indian) — do not add more
- Urals interim data is acceptable for training; must be replaced before paper submission
- All decisions must be logged in PROJECT_STATUS.md

## Deviation Alerts
If any agent proposes:
- Changing the 5-node architecture → REJECT, flag to user
- Adding a 6th benchmark → REJECT unless user explicitly approves
- Skipping ablation experiments → REJECT
- Submitting without real Urals data → REJECT

## Key Files
- `PROJECT_STATUS.md` — Phase tracker, decisions log
- `DATA_STATUS.md` — Per-dataset status
- `ARCHITECTURE.md` — Model spec (single source of truth)
- `src/config.py` — All hyperparameters and paths
