# GeoRipNet — Agent System

This directory contains instruction files for Claude Code agents working on this project.
Each file defines one agent's role, rules, and current state.

## How to Use

When starting a new Claude Code session, reference the relevant agent file:
- **Starting work on data**: "Act as the Data Agent per `agents/DATA_AGENT.md`"
- **Starting work on model code**: "Act as the Code Agent per `agents/CODE_AGENT.md`"
- **Reviewing changes**: "Act as the Review Agent per `agents/REVIEW_AGENT.md`"
- **Writing documentation**: "Act as the Docs Agent per `agents/DOCS_AGENT.md`"
- **Checking project direction**: "Check `agents/ORCHESTRATOR.md` for current phase and constraints"

## Agent Files

| File | Role |
|------|------|
| `ORCHESTRATOR.md` | Project coordination, phase tracking, constraint enforcement |
| `DATA_AGENT.md` | Data collection, validation, pipeline |
| `CODE_AGENT.md` | Model code, training, evaluation |
| `REVIEW_AGENT.md` | Code review, methodology validation |
| `DOCS_AGENT.md` | Paper writing, documentation |

## Quick Status Check

Read `PROJECT_STATUS.md` (root) for overall progress.
Read `DATA_STATUS.md` (root) for data readiness.

## Non-Negotiable Constraints
- 5 nodes only: WTI, Brent, OPEC Basket, Urals, Indian Basket
- Training: 2010-2019 | Val: 2020-2021 | Test: 2022-present
- Target journal: Energy Economics (Elsevier, IF ~13)
- All hyperparameters in `src/config.py` — nowhere else
