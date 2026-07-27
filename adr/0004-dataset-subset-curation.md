# ADR-0004: Fixed curated subset instead of full corpus

Status: Accepted
Date: 2026-07-24

## Context

Dataset source is Ego-Exo4D (corrected from an earlier assumption of EgoDex — see
[things-i-learned.md](../things-i-learned.md)), which is large and egocentric+exocentric. Data
volume is not the constraint here — rented GPU time is. Training all 4 strategies on the full
corpus would blow the compute budget for no clear benefit to the research question.

## Decision

Build one fixed, curated subset (diverse tasks / interaction types, egocentric views only) once,
split into train/held-out-eval, and reuse that exact split identically across all 4 strategy
runs (A–D). Curation and split happen before any training run, not per-run.

## Consequences

- Comparisons between strategies stay apples-to-apples — a score difference can be attributed to
  the adaptation strategy, not a data difference.
- Subset size must be picked to balance statistical meaningfulness against per-run training
  cost — sized against the budget in the PRD (§9), not against "how much data is available."
- Curation criteria (task diversity, interaction-type balance) need to be written down
  explicitly before building the split, so the process is reproducible and defensible in the
  writeup.
