# ADR-0002: Adaptation strategy set (partial tuning)

Status: Accepted
Date: 2026-07-24

## Context

Original brainstorm (from Codex conversation, see `codex_suggestions.md`) proposed 8 strategies
(A–H: full LoRA, vision-only, language-only, projector-only, attention-only, attention+MLP, top-4
decoder blocks, hybrid). With zero local GPU and a rented-by-the-hour budget, 8 full training +
eval runs is not affordable, and "projector-only" support in Unsloth for the chosen model is
unverified.

## Decision

Trim to 4 strategies that still answer the research question (where does the egocentric domain
gap live — vision or language?) plus a cheap baseline and a middle ground:

| # | Strategy | Unsloth config |
|---|----------|-----------------|
| A | Full LoRA (baseline) | `finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True` |
| B | Vision-encoder only | `finetune_vision_layers=True`, rest `False` |
| C | Language-decoder only | `finetune_language_layers=True`, rest `False` |
| D | Attention + MLP hybrid | `finetune_attention_modules=True, finetune_mlp_modules=True`, vision/language layer flags `False` |

## Consequences

- Each strategy is implemented as a named config in `train.py`, not a separate script
  (see [ADR](0002-adaptation-strategy-set.md) code-structure note below and `tasks.md`).
- Dropped strategies (projector-only, attention-only, top-4-decoder-blocks) are explicitly
  out of scope. If budget remains after A–D, projector-only is the first candidate to add back
  (directly tests the modality-bridging hypothesis) — not a full return to 8.
- All 4 runs MUST share identical data split, seed, steps, and batch size — only the adaptation
  config differs. Implemented via a single `train.py --strategy {A,B,C,D}` entrypoint so this
  can't silently drift.
