# EgoScribe

**Given a fixed, small compute budget, which parts of a pretrained VLM should you adapt to
improve egocentric hand-object-interaction understanding — the vision encoder, the language
decoder, attention vs. MLP, or the projector?**

This is a controlled parameter-efficient adaptation study, not a "I fine-tuned a VLM" demo. It
takes Qwen2.5-VL-7B-Instruct, trains four QLoRA adaptation strategies on the same curated
Ego-Exo4D subset, and evaluates them honestly on a held-out split — including reporting the
cases where the data does **not** support a conclusion.

Whole study ran on **$16.25 of GCP compute** (inside a free-trial credit, $0 out of pocket), no
local GPU.

## The four strategies

Same base model, split, seed, and 3-epoch recipe for all four. Only *where* the LoRA adapter
lives changes.

| # | Strategy | What it isolates | Trainable params |
|---|----------|------------------|---:|
| A | Vision + language, attention + MLP | Broad adaptation / upper bound | 51.5M (0.62%) |
| B | Vision only, attention + MLP | Does the domain gap live in *seeing*? | 11.2M (0.13%) |
| C | Language only, attention + MLP | Does it live in *describing/reasoning*? | 40.4M (0.48%) |
| D | Both modalities, attention only | Cheaper middle ground, no MLP | 14.0M (0.17%) |

## What we found

Token-F1 with 95% take-level bootstrap intervals over the six held-out takes (the intervals are
wide on purpose — six takes is a small independent-unit count). Full table and reasoning in
[`docs/results_summary.md`](docs/results_summary.md); source data in [`docs/eval_out/`](docs/eval_out/).

- **A and C are statistically indistinguishable** on every semantic field. No "C beats A" claim
  is made — the intervals overlap almost completely.
- **Vision-only (B) is the weakest** on the language-shaped fields (action-verb, current-state) —
  the closest thing to a real separation in the study.
- **Point-of-no-return F1 is uninformative** (intervals span ~[0, 0.65] on five positives).
- The **defensible, noise-robust finding is the efficiency one:** language-only adaptation (C)
  matches full vision+language adaptation (A) at fewer trainable params, while vision-only (B)
  underperforms — pointing to the domain gap living on the description side, not the seeing side.

This is an **exploratory** result over six takes, not a definitive ranking. The honest framing —
recognizing that 110 segments are really six correlated takes, switching to a take-level
bootstrap, and refusing to overclaim overlapping intervals — is the point of the study as much as
any single number.

Not measurable here: `safety_gear_missing` (zero positive gold items in the held-out split) —
reported as *unavailable*, not solved.

## Repo layout

- `src/data_prep/` — Ego-Exo4D → schema conversion, LLM-assisted labeling, curated-subset split.
- `src/dataset.py` — `EgocentricHOIDataset`; single source of truth for the training prompt.
- `src/schema.py` — `HandObjectInteraction`, the structured output contract.
- `train.py` — QLoRA training, `--strategy {A,B,C,D}` (see `adr/`).
- `src/eval/` — evaluation harness: schema validity, per-field F1, take-level bootstrap CIs.
- `serve.py` — one adapter-backed inference example; reuses the eval loader/prompt/validation.
- `docs/results_summary.md`, `docs/eval_out/` — final numbers and the raw clustered outputs.
- `adr/`, `PRD.md`, `tasks.md`, `things-i-learned.md` — decisions, scope, and what broke.

## Run

Training (per strategy):

```
python train.py --strategy A --annotations data/converted/annotations_train.json
```

Evaluation with take-level CIs:

```
python -m src.eval.evaluate --adapter saved_egoscribe_adapters/strategy_A \
  --annotations data/converted/annotations_eval.json
```

Inference against a trained checkpoint (GPU) — one held-out example:

```
python serve.py --annotations data/converted/annotations_eval.json \
  --video_dir data/ego-exo4d/takes --adapter saved_egoscribe_adapters/strategy_C --index 0
```

## Honest limitations

- Six held-out takes → wide intervals; comparisons are exploratory, not conclusive.
- Absolute semantic F1 is low; small data (572 train segments) and 3 epochs likely undertrain.
- Token-F1 on text fields structurally favors language-side adaptation — a confound on the
  "adapt language, not vision" reading, not independent proof of it.
- The trained-checkpoint serving path is code-ready but not yet run end-to-end on a GPU.

See [`things-i-learned.md`](things-i-learned.md) for the bugs and course-corrections along the way.
