# ADR-0005: Compute provider for Phase 2 real strategy runs

Status: Accepted
Date: 2026-08-06

## Context

Phase 0 (pipeline validation) ran free on Colab T4. Phase 2 (real training runs of all 4
strategies on the curated Ego-Exo4D subset) needs a real GPU with actual bf16 tensor core
support (T4 doesn't have this — see [things-i-learned.md](../things-i-learned.md) for the bug
this caused during the smoke test).

Compared three options on price and access, pulled fresh rather than assumed (pricing in this
market moves fast):

| Provider | GPU | Rate | Notes |
|---|---|---|---|
| Vast.ai | RTX 4090 (24GB) | ~$0.29-0.39/hr spot | Cheapest, but spot = preemption risk |
| RunPod | RTX 4090 (24GB) | ~$0.34-0.69/hr | More reliable than Vast.ai spot |
| Google Cloud | L4 (24GB) | $0.70/hr on-demand | No spot-market volatility |

## Decision

**Google Cloud, G2 instance with 1x NVIDIA L4 (24GB)**, paid from a $300/90-day free-trial
credit — not the cheapest per-hour rate, but the estimated total training cost (rough math:
20-40hrs total across all 4 strategies, §9 of the PRD) is well under $300, making Phase 2
likely free in practice, which beats optimizing per-hour rate on Vast.ai/RunPod.

Tradeoff accepted: GCP's free-trial billing account cannot attach a GPU at all — upgrading to a
paid billing account (real card on file) is required to unlock GPU access. No charge occurs
unless the $300 credit is exceeded.

## Consequences

- Training-time/cost estimates in the PRD are still a rough extrapolation from the T4 smoke
  test — plan is to benchmark Strategy A for ~15-20 min on the real L4 first, before committing
  to full runs of all 4 strategies (tracked in tasks.md, Phase 2).
- If the $300 credit runs out before all 4 strategies + any re-runs are done, fall back to
  Vast.ai/RunPod RTX 4090 for the remainder — same QLoRA setup (ADR-0003) fits either GPU.
- Worth periodically checking actual GCP spend against the credit rather than assuming it covers
  everything indefinitely.
