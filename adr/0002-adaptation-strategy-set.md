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
gap live — vision or language?) plus a cheap baseline and a middle ground.

**Correction (2026-08-XX, during Colab smoke test):** the original table below set B's module-
type flags (attention/MLP) both `False` and D's layer-scope flags (vision/language) both
`False`. Unsloth's 4 switches are two independent axes — WHICH layers (vision/language) ×
WHAT within them (attention/MLP) — and Unsloth raises `RuntimeError: No modules to finetune`
if either whole axis is all-`False`, regardless of the other axis. Both B and D were invalid
configs that would target nothing. Fixed table:

| # | Strategy | Unsloth config |
|---|----------|-----------------|
| A | Full LoRA (baseline) | `finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True` |
| B | Vision-encoder only | `finetune_vision_layers=True, finetune_language_layers=False, finetune_attention_modules=True, finetune_mlp_modules=True` |
| C | Language-decoder only | `finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True` |
| D | Attention-only, both modalities | `finetune_vision_layers=True, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=False` |

D's original description ("attention + MLP hybrid... cheaper than A") was also self-contradictory
once corrected — if D targets both vision+language layers AND both attention+MLP module types,
it's identical to A, not cheaper. Redefined D as attention-only across both modalities: fewer
trainable params than A (no MLP), but broader scope than B/C (spans both layer types) — a
genuinely distinct fourth point in the comparison.

## Interpretation (how far the 4 arms can be pushed)

The 4 arms are NOT a clean factorial and must not be written up as one ranked ablation. They
answer overlapping questions:

- **B vs C** — vision vs language adaptation (the core research question).
- **A vs D** — incremental value of MLP adapters vs attention-only. This is D's specific job:
  test whether attention-only LoRA across both vision and language layers captures most of the
  benefit of full broad LoRA.
- **A vs B/C/D** — broad vs restricted adaptation.

Caveats that bound every claim, D's especially:

- **Param-count confound.** A/B/C/D differ in trainable-parameter *count* as well as location,
  so the A–D contrast is confounded with parameter count — report it as *suggestive*, not as a
  clean isolation of "MLP adapters."
- **"Cheapest" means parameter-efficient, not runtime-cheaper.** Frozen layers still run the
  forward pass; the smoke test already showed B's wall-time gain over A was modest. Actual
  cost/step must be *measured* on the L4, never inferred from param count.
- **The statistical unit is the take, not the segment.** Eval is 110 segments but only 6
  held-out takes; segments from one take are correlated. Metrics must be aggregated and
  bootstrapped at the take level (see `src/eval/metrics.py`), and at n=6 takes the CIs are wide
  by construction — treat them as an honesty instrument, not a significance test.
- **Literature prior is motivation, not evidence.** "Attention-only often captures most gains"
  justifies running D; it does not establish the result for Qwen2.5-VL on egocentric video with
  these weak labels.

Standing writeup framing for D:

> Strategy D tests whether attention-only LoRA across both vision and language layers provides
> most of the benefit of full broad LoRA. The A–D contrast estimates the incremental value of
> adding MLP adapters, but is confounded with trainable parameter count and should be
> interpreted as suggestive.

## Consequences

- Each strategy is implemented as a named config in `train.py`, not a separate script
  (see [ADR](0002-adaptation-strategy-set.md) code-structure note below and `tasks.md`).
- Dropped strategies (projector-only, attention-only, top-4-decoder-blocks) are explicitly
  out of scope. If budget remains after A–D, projector-only is the first candidate to add back
  (directly tests the modality-bridging hypothesis) — not a full return to 8.
- All 4 runs MUST share identical data split, seed, steps, and batch size — only the adaptation
  config differs. Implemented via a single `train.py --strategy {A,B,C,D}` entrypoint so this
  can't silently drift.
