# PRD — EgoScribe: Compute-Efficient Adaptation of VLMs for Egocentric Procedural Understanding

Status: In Progress — Phases 0–2, evaluation, and analysis/writeup complete (results summary with
take-level CIs, cost recorded, README shipped). Remaining: end-to-end GPU serving run of a trained
checkpoint (serve.py is code-ready).
Last updated: 2026-08-17

## 1. Problem

Off-the-shelf VLMs (Qwen2.5-VL-7B-Instruct — see [ADR-0001](adr/0001-model-selection.md), model
choice now settled) are not trained on first-person, hand-object-interaction video. Full
fine-tuning is out of reach on a student/no-GPU budget. The naive answer ("apply LoRA
everywhere") is also just an assumption — nobody has shown that's the right place to spend a
limited parameter/compute budget for this domain.

## 2. Research question

> Given a fixed, small compute budget, which components of a pretrained VLM (vision encoder,
> language decoder, attention vs. MLP, or the modality projector) should be adapted to maximize
> performance on egocentric hand-object-interaction understanding?

This reframes the project from "I fine-tuned a VLM" to an empirical ablation study with a
defensible conclusion — see [things-i-learned.md](things-i-learned.md) for how this framing
came about.

## 3. Goals

- Produce a controlled comparison of a small set of parameter-efficient adaptation strategies
  (see [ADR-0002](adr/0002-adaptation-strategy-set.md)) on a fixed, curated Ego-Exo4D subset.
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

- Source: **Ego-Exo4D** (Meta/FAIR). License granted, credentials received, real data
  downloaded — no longer a plan, this is done (correcting an earlier assumption that this was
  EgoDex; the two have different schemas/access terms — see
  [things-i-learned.md](things-i-learned.md)).
- We only use the **egocentric** (aria/rgb) views; exocentric cameras are out of scope, matching
  the "first-person" framing of the research question.
- **Curated subset built and validated** (`src/data_prep/`): 3 deliberately distinct interaction
  domains chosen from 17 available scenarios — Covid-19 Rapid Antigen Test (medical,
  fine-motor), Fix a Flat Tire (mechanical, tool use), Cooking an Omelet (heat/state-change) —
  10 takes each, 30 total. Full rationale in [ADR-0004](adr/0004-dataset-subset-curation.md).
- Labels come from `src/data_prep/llm_label_segments.py` (Claude Haiku 4.5, cross-referencing
  keystep segments against `atomic_descriptions` narrations for richer extraction context) —
  a large quality improvement over the naive step_name-split heuristic it replaces (~40-50%
  clean vs. LLM's consistently clean output on spot-checks). Measured cost: ~$0.0019/segment,
  ~$1.27 for the full curated subset (682 segments).
- Video: downloaded via `egoexo --parts downscaled_takes/448` (448px pre-downscaled variant,
  ~4GB for the 30 curated takes) — not full resolution (~80GB+ for the same 30 takes), since
  `EgocentricHOIDataset` already caps resolution hard.
- `data/converted/annotations_train.json` (572 segments / 24 takes) and
  `annotations_eval.json` (110 segments / 6 takes) are the real curated dataset — split by
  `take_uid` (not segment, to avoid context leakage), stratified per domain (8/2 takes each),
  fixed seed. This exact pair is reused identically across all 4 strategy runs.
  (`src/data_prep/split_curated_dataset.py`)
- `data/samples/` (`0.mp4`, `1.mp4`, `4.mp4`) is the original hand-built toy set — kept only for
  pipeline smoke-testing (Phase 0), not used for real training.
- All labels spot-checked against actual video frames (3 samples, one per domain) — confirmed
  accurate, including one detail (a "lighter" tool) verified by manually scrubbing the source
  video after a single still frame didn't show it clearly.

## 6. Model

**Settled**: `Qwen2.5-VL-7B-Instruct`, loaded via `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`
(Unsloth's Dynamic 4-bit quant) for training. See [ADR-0001](adr/0001-model-selection.md) —
Accepted. `serve.py` already updated to match (base `Qwen/Qwen2.5-VL-7B-Instruct` checkpoint via
SGLang, not the training quant — SGLang expects a different format).

## 7. Method — adaptation strategies

Trimmed from an original 8-way grid to 4, to fit budget while still answering the research
question. Full rationale in [ADR-0002](adr/0002-adaptation-strategy-set.md).

| # | Strategy | Tests |
|---|----------|-------|
| A | Full LoRA (attention, vision, language) | Baseline / upper bound on trainable params |
| B | Vision-encoder only (attention + MLP, scoped to vision) | Does the domain gap live in *seeing* egocentric frames? |
| C | Language-decoder only (attention + MLP, scoped to language) | Does it live in describing/reasoning over what's seen? |
| D | Attention-only, both modalities (no MLP) | Middle ground — cheaper than A, broader than B/C |

B/C/D configs were corrected mid-project after a real bug: Unsloth's 4 finetune switches are
two independent axes (which layers × which module types), not 4 independent toggles — the
original B/D configs each zeroed out a whole axis and would have trained nothing. Caught on the
Colab smoke test, not in production. See ADR-0002 and
[things-i-learned.md](things-i-learned.md).

All four trained with QLoRA (4-bit) via Unsloth — see [ADR-0003](adr/0003-quantization-qlora.md).
All 4 smoke-tested clean with distinct trainable-param counts (A=51.5M/0.62%, B=11.2M/0.13%,
C=40.4M/0.48%, D=14.0M/0.17%) — config correctness confirmed before any real-data run.

## 8. Evaluation

Held-out split from the same curated subset (never seen in any of the 4 training runs).
Metrics, all derived from `HandObjectInteraction` schema fields:
- JSON schema validity rate (does output even parse?)
- Per-field accuracy: `tool_detected`, `target_object`, `action_verb`, `current_state`
- `point_of_no_return_detected` — binary F1 (this is the "state change" moment, likely the
  hardest field)
- `safety_gear_missing` — set-level precision/recall

The evaluation harness and all four evaluations are complete. A/B/C were originally evaluated
before take-level bootstrap support landed, then rescored offline from their saved prediction
dumps. D was evaluated with the bootstrap-aware harness. The final `*_clustered.json` outputs
report point estimates and 95% block-bootstrap intervals over the six held-out takes; the takes,
not the 110 segments, are the independent statistical units.

Results are exploratory: A, B, and C produced valid JSON for all 110 held-out segments; D
produced valid JSON for 101/110. The take-level bootstrap intervals are wide and overlap: A and
C are statistically indistinguishable on every semantic field, so no A-vs-C ranking is claimed.
The one clear trend is that vision-only (B) is weakest on the language-shaped fields (action,
state), while language-only (C) matches full adaptation (A) at fewer params — pointing to the
domain gap living on the description side, not the seeing side. PONR F1 is uninformative
(intervals span ~[0,0.65] on five positives). The `safety_gear_missing` metric is not measurable
in this split because the gold labels contain zero positive safety-gear items. Full per-field
intervals in [docs/results_summary.md](docs/results_summary.md); source data committed under
`docs/eval_out/`.

## 9. Compute & budget

- No local GPU. Phase 0 pipeline validation (smoke test, all 4 strategies) done for free on
  Colab T4 — confirmed the training loop, collator, and all 4 strategy configs work correctly
  before spending any money.
- **Phase 2 (real runs) provider: Google Cloud, G2 instance with 1x NVIDIA L4 (24GB)**, paid via
  a $300/90-day free-trial credit — chosen over Vast.ai/RunPod RTX 4090 (~$0.30-0.70/hr,
  cheaper per-hour but real money) because the estimated total training cost is well under the
  credit, making Phase 2 likely free. Tradeoff: GCP requires upgrading to a paid billing account
  (real card on file) to unlock GPU access at all — no charge unless the credit is exceeded.
- `train.py` already loads the model via Unsloth 4-bit QLoRA (`unsloth-bnb-4bit`), not full
  BF16 — this was the very first Phase 0 fix, done before any smoke test.
- The real L4 runs validated the runtime estimate: each three-epoch strategy took roughly
  2.4–2.7 hours. **Total actual GCP spend: $16.25** (Compute Engine $16.19 + Networking $0.06)
  from the billing console — fully covered by the $300 free-trial credit, $0 out of pocket. ~$7
  of that is the four training runs at list price (~$1.7–1.9 each); the rest is VM overhead
  (setup, transfer, eval, idle). This matched the earlier caution: the billing number is ~2x a
  training-only estimate inferred from elapsed time, so billing was the right source.

## 10. Risks

Resolved during Phase 0/1 (kept here for record, not because they're still open):
- ~~Ego-Exo4D → schema conversion nontrivial~~ — done. Conversion script built, tested against
  real data, found and fixed 3 real bugs along the way (duplicated path segment, wrong-file
  scenario field, unusable-take filtering) — genuinely took real effort, as predicted, but
  didn't blow the budget.
- ~~Gated access~~ — license granted, credentials received, data downloaded and verified.
- ~~Label masking bug~~ — fixed in Phase 0, before any training run.

Still open:
- **Small eval set (110 segments / 6 takes) → noisy comparisons** between strategies. Mitigate:
  report the numbers honestly in the writeup, don't oversell small deltas as significant.
- **Scope creep back to 8 strategies** — explicitly rejected in ADR-0002; revisit only if a
  strategy pair (A/B/C/D) is ambiguous and budget remains.
- ~~Exact cloud cost is not yet recorded~~ — recorded: $16.25 total from billing (see §9),
  within the $300 credit. Confirmed the billing figure is ~2x a training-only elapsed-time
  estimate, so billing was the right source.

## 11. Milestones

See [tasks.md](tasks.md) for the working checklist.

- **Phase 0 — Setup & fixes**: done. All code bugs found via Colab smoke test fixed; all 4
  strategies confirmed correct before spending on real data.
- **Phase 1 — Data**: done. Real Ego-Exo4D curated subset built, LLM-labeled, converted,
  spot-checked, split into train/eval.
- **Phase 2 — Real strategy runs (A/B/C/D)**: done. All four trained on the identical
  take-disjoint split and their checkpoints were backed up locally.
- **Phase 4 — Evaluation**: done. All four checkpoints were evaluated; final outputs include
  take-level bootstrap confidence intervals.
- **Phase 5 — Analysis & writeup**: done except the GPU serving run. Comparison table with
  take-level CIs, parameter efficiency, and cost ($16.25 total) are in
  [docs/results_summary.md](docs/results_summary.md); README shipped; limitations documented.
  `serve.py` is adapter-capable but the end-to-end checkpoint run on a GPU is still open.
