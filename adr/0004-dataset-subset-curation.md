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

## Curation criteria (2026-08-XX)

`keystep_train.json` covers 668 takes across 17 scenarios (surveyed directly, not estimated):
Covid-19 Rapid Antigen Test (157), Fix a Flat Tire (71), Remove/Install a Wheel (56/53),
cooking scenarios — omelet, salads, noodles, eggs (spanning ~20-48 each), First Aid/CPR (43),
beverage-making — milk tea, coffee, chai (11-33 each), plus a few single-digit scenarios
(pasta, sushi).

**Selected: 3 scenarios spanning distinct interaction domains**, not picked by take-count:
- **Medical**: Covid-19 Rapid Antigen Test — fine-motor, small-object manipulation, no tools
- **Mechanical**: Fix a Flat Tire — tool use (levers, pumps), larger/coarser motion
- **Cooking**: Cooking an Omelet — different tool set (utensils), heat/state-change focus

Rationale: these three differ enough in tool use, motion scale, and what a "point of no return"
looks like (breaking a test seal vs. removing a tire bead vs. cracking an egg) that if a
strategy's ranking holds across all three, that's a real signal, not an artifact of one domain.
Deliberately excluded from the curated set (not because they're bad data, just kept out of
scope for this pass): CPR (safety-critical labeling requires more care than budget allows for
this project), all beverage/other cooking scenarios (redundant with the omelet scenario for
domain coverage).

**Sizing**: ~8-10 takes per scenario (~24-30 takes total), split by `take_uid` (not by segment)
into train/held-out-eval — splitting by take, not segment, avoids leaking context from the same
video into both train and eval. Exact per-scenario counts and the split ratio are set when the
curation script runs (`src/data_prep/convert_egoexo4d.py --scenario`), not fixed in this ADR.

**Labeling cost**: LLM-assisted labeling (`src/data_prep/llm_label_segments.py`) costs ~$0.0019/
segment at Haiku 4.5 rates (measured, not estimated, on a 100-segment sample). At ~20 segments/
take, ~30 takes ≈ 600 segments ≈ ~$1.15 — labeling only the curated subset instead of the full
corpus (~$25 for all 668 takes) is the direct payoff of curating first.
