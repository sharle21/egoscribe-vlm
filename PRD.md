# PRD — EgoScribe: Compute-Efficient Adaptation of VLMs for Egocentric Procedural Understanding

Status: Draft
Last updated: 2026-07-23

## 1. Problem

Off-the-shelf VLMs (Qwen2.5-VL / MiMo-VL family — see [ADR-0001](adr/0001-model-selection.md)) are not
trained on first-person, hand-object-interaction video. Full fine-tuning is out of reach on a
student/no-GPU budget. The naive answer ("apply LoRA everywhere") is also just an assumption —
nobody has shown that's the right place to spend a limited parameter/compute budget for this
domain.

## 2. Research question

> Given a fixed, small compute budget, which components of a pretrained VLM (vision encoder,
> language decoder, attention vs. MLP, or the modality projector) should be adapted to maximize
> performance on egocentric hand-object-interaction understanding?

This reframes the project from "I fine-tuned a VLM" to an empirical ablation study with a
defensible conclusion — see [things-i-learned.md](things-i-learned.md) for how this framing
came about.

## 3. Goals

- Produce a controlled comparison of a small set of parameter-efficient adaptation strategies
  (see [ADR-0002](adr/0002-adaptation-strategy-set.md)) on a fixed, curated EgoDex subset.
- Every run trainable on a single rented GPU (≤24–48GB) within a $20–40 total project budget.
- Ship a working inference pipeline (`serve.py`) that outputs structured JSON
  (`src/schema.py::HandObjectInteraction`) from raw egocentric clips.
- Produce an honest writeup: what worked, what didn't, and why — not a hype README.

## 4. Non-goals

- Not attempting SOTA on any public benchmark.
- Not fine-tuning on the full Ego-Exo4D corpus — a curated subset is enough to answer the
  research question (see [ADR-0004](adr/0004-dataset-subset-curation.md)).
- Not building a general-purpose video-VLM training framework — this is a scoped study, not a
  library.
- Not doing full (non-LoRA) fine-tuning of any strategy.

## 5. Dataset

- Source: **Ego-Exo4D** (Meta/FAIR), large-scale egocentric + exocentric video of skilled human
  activities, with narrations/keystep annotations. Gated dataset — requires signed license
  agreement, not a plain HuggingFace download (correcting earlier assumption that this was
  EgoDex; the two have different annotation schemas and access terms — see
  [things-i-learned.md](things-i-learned.md)).
- We only use the **egocentric** views; exocentric (third-person) views are out of scope for
  this project (matches the "first-person" framing of the research question).
- Current `data/samples/` (`0.mp4`, `1.mp4`, `4.mp4`) is a hand-built toy set only — not
  representative, used to validate the pipeline shape.
- `src/dataset.py::EgocentricHOIDataset` expects a custom annotation schema
  (`interaction_start_frame`, `interaction_end_frame`, `expected_output`) that raw Ego-Exo4D
  annotations do NOT match — a conversion step is required before real data can be loaded
  (tracked in [tasks.md](tasks.md)). Ego-Exo4D's native annotations are narration/keystep-based,
  not frame-range hand-object-interaction labels, so this conversion is more involved than a
  simple field remap — likely needs a heuristic or model-assisted step to derive
  interaction start/end frames from keystep timestamps.
- Because Ego-Exo4D is large, data volume is not the constraint — subset *curation* is. One
  fixed subset (diverse tasks / interaction types, held-out eval split) is built once and reused
  identically across every strategy run, so comparisons are apples-to-apples.

## 6. Model

Open decision — `train.py` currently references `XiaomiMiMo/MiMo-VL-7B-RL` while `serve.py`
also targets it, but the training code instantiates it via
`Qwen2_5_VLForConditionalGeneration` (MiMo-VL is built on the Qwen2.5-VL backbone, so this
mostly works, but the choice hasn't been deliberately made). Tracked in
[ADR-0001](adr/0001-model-selection.md) — resolve before Phase 2 (baseline run).

## 7. Method — adaptation strategies

Trimmed from an original 8-way grid to 4, to fit budget while still answering the research
question. Full rationale in [ADR-0002](adr/0002-adaptation-strategy-set.md).

| # | Strategy | Tests |
|---|----------|-------|
| A | Full LoRA (attention, vision, language) | Baseline / upper bound on trainable params |
| B | Vision-encoder only | Does the domain gap live in *seeing* egocentric frames? |
| C | Language-decoder only | Does it live in describing/reasoning over what's seen? |
| D | Attention + MLP (hybrid, best-supported combo in Unsloth for the chosen model) | Middle ground — cheaper than A, more expressive than B/C |

All four trained with QLoRA (4-bit) via Unsloth — see
[ADR-0003](adr/0003-quantization-qlora.md).

## 8. Evaluation

Held-out split from the same curated subset (never seen in any of the 4 training runs).
Metrics, all derived from `HandObjectInteraction` schema fields:
- JSON schema validity rate (does output even parse?)
- Per-field accuracy: `tool_detected`, `target_object`, `action_verb`, `current_state`
- `point_of_no_return_detected` — binary F1 (this is the "state change" moment, likely the
  hardest field)
- `safety_gear_missing` — set-level precision/recall

## 9. Compute & budget

- No local GPU. Rent spot instances (RunPod / Vast.ai). RTX 4090 (24GB) ≈ $0.30–0.50/hr,
  A6000 (48GB) ≈ $0.50–0.80/hr.
- Prototype pipeline correctness for free first (Colab T4 / Kaggle 30 free GPU-hrs/week) before
  spending money on real runs.
- Target: each of the 4 strategy runs ≤ $5, total project compute budget ≤ $30–40.
- `train.py` currently loads the model in full `bfloat16`, not 4-bit — must switch to
  `bitsandbytes`/Unsloth 4-bit loading before any paid run (tracked in tasks.md) — full BF16
  won't fit a free-tier T4 and eats most of a 4090's headroom.

## 10. Risks

- **Ego-Exo4D → custom schema conversion is nontrivial** (narration/keystep timestamps, not
  frame-range HOI labels) and easy to underestimate → could eat the whole budget before training
  starts. Mitigate: timebox conversion, fall back to a smaller manually reviewed subset if
  needed.
- **Gated access** — Ego-Exo4D requires a signed license agreement; confirm access is actually
  granted before planning around it as the data source.
- **Small eval set → noisy comparisons** between strategies. Mitigate: report confidence
  intervals / multiple seeds if budget allows, be honest in the writeup if it doesn't.
- **Label masking bug** in current dataset loader (padding tokens included in loss) will bias
  early comparisons if not fixed before baseline run.
- **Scope creep back to 8 strategies** — explicitly rejected in ADR-0002; revisit only if a
  strategy pair (A/B/C/D) is ambiguous and budget remains.

## 11. Milestones

See [tasks.md](tasks.md) for the working checklist. Rough phases: Setup & fixes → Data
conversion & curation → Baseline (Strategy A) → Strategies B/C/D → Eval harness → Analysis &
writeup.
