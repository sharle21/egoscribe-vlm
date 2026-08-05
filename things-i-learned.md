# things-i-learned.md — interview prep notes

Format: dated entry, what happened, what I learned, how I'd say it in an interview.

---

## 2026-07-23 — Reframing the project around a research question, not a technique

**What happened:** Started as "fine-tune Qwen2.5-VL with LoRA." A conversation with Codex
suggested reframing around compute-efficient adaptation as an empirical question.

**What I learned:** "I fine-tuned model X" is not a story. "I had a fixed compute budget and had
to decide where to spend it, so I tested it" is. The constraint became the thesis instead of
something to apologize for.

**Interview answer:** "I didn't assume LoRA-everywhere was optimal for our domain gap. I had a
compute budget, so I set up a controlled comparison of which model components actually needed
adaptation for egocentric hand-object-interaction understanding, and picked the answer that
generalized best per dollar of GPU time spent."

---

## 2026-07-23 — Scope: 8 ablation strategies was the wrong number for the budget

**What happened:** Initial brainstorm produced 8 partial-tuning strategies. On reflection, each
strategy is a full train+eval run — with zero local GPU and a rented budget, 8 runs plus a
consistent eval harness is not "cheap" just because each individual run is cheaper with Unsloth.

**What I learned:** Tooling that makes each experiment cheaper doesn't mean you should run more
experiments — it means you can afford to run the *right few* well, with proper eval. Trimmed to
4 (full LoRA baseline, vision-only, language-only, attention+MLP hybrid) — see
[ADR-0002](adr/0002-adaptation-strategy-set.md).

**Interview answer:** "I initially scoped 8 conditions, then realized the eval harness cost more
than the training runs. I cut to 4 that still isolated the vision-vs-language question, which
kept the study affordable without losing the core finding."

---

## 2026-07-24 — Mixed up two different egocentric datasets (EgoDex vs Ego-Exo4D)

**What happened:** Planned around "EgoDex" from HuggingFace before realizing the actual dataset
in scope is Ego-Exo4D (Meta/FAIR) — a different dataset with different annotation schema
(narration/keystep timestamps, not frame-range hand-object-interaction labels) and gated
access requiring a signed license, not a plain download.

**What I learned:** Don't design a data-conversion plan around a dataset's *name* before
confirming its actual schema and access terms — the two egocentric datasets sound similar but
aren't interchangeable, and the mistake would have shown up expensively (mid-conversion-script)
rather than cheaply (right now, on paper).

**Interview answer:** "I caught a dataset mix-up during planning, before writing any conversion
code — worth flagging because it's the kind of mistake that's cheap to fix on paper and
expensive to fix after you've built a pipeline around the wrong schema."

---

## 2026-08-05 — Unsloth's partial-tuning switches are two axes, not four independent toggles

**What happened:** Designed 4 ablation strategies (ADR-0002) assuming
`finetune_vision_layers`/`finetune_language_layers`/`finetune_attention_modules`/
`finetune_mlp_modules` were 4 independent on/off flags. Strategy B ("vision only") set
`finetune_vision_layers=True` and left the other 3 `False` — including both module-type flags.
On the Colab smoke test this raised `RuntimeError: No modules to finetune`. Strategy D had the
mirror-image bug (both layer-scope flags `False`).

**What I learned:** The 4 flags are really 2 independent axes — WHICH layers (vision/language)
crossed with WHAT within them (attention/MLP modules) — and at least one flag on *each* axis
must be `True`, or there's nothing to select regardless of the other axis. "Vision only" isn't
`vision_layers=True, everything else False` — it's `vision_layers=True, language_layers=False,
attention_modules=True, mlp_modules=True` (full expressiveness, scoped to vision). This also
meant D's original definition was internally contradictory once I fixed the semantics — I
redefined it as attention-only across both modalities, a real fourth point instead of
duplicating A.

**Interview answer:** "I designed a 2x2-style config space without realizing it was actually
2x2 — I'd mentally modeled 4 independent switches. Running it against the real API surfaced
the error immediately: two of my four planned strategies were malformed and would have
silently trained nothing, or in D's case, secretly duplicated the baseline. It's a good example
of why I run configs against the real training loop early, on a toy dataset, before trusting
the design on paper."

---

## 2026-08-05 — A config file that silently does nothing (accelerate launch vs plain python)

**What happened:** Strategy C crashed with a dtype mismatch (`mat1 and mat2 must have the same
dtype, but got Float and BFloat16`) deep in the frozen vision encoder's attention layer, only
during the backward pass. First two fix attempts were reasonable but wrong: (1) pinning an
explicit `dtype` on model load, (2) disabling gradient checkpointing (the crash happened inside
its recompute path, so it looked implicated). Neither fixed it — same error, same line, twice.
Root cause: `config/accelerate_config.yaml` sets `mixed_precision: bf16`, but that file is only
read when the script is launched via `accelerate launch --config_file ...`. The actual command
being run was plain `python train.py` (a Colab cell) — so `Accelerator()` silently fell back to
its own default (`mixed_precision: no`), and the config file did nothing at all. Without
autocast, nothing forces consistent dtype outside of LoRA-wrapped layers (PEFT casts its own
inputs internally on the layers it wraps) — Strategy A and B "worked" by accident, because both
happened to LoRA-wrap the vision encoder; Strategy C was the first to leave it fully frozen,
exposing the gap.

**What I learned:** A correctly-written config file is not the same as a config file that's
actually being read. When a fix "should" work and doesn't, check whether the mechanism meant to
apply it is even in the execution path — not just whether the setting itself is correct. Two
failed fixes in a row on the same error is itself a signal to step back and question the whole
diagnosis, not just try a third variant of the same theory.

**Interview answer:** "A crash pointed at gradient checkpointing, and I spent two fix attempts
chasing that theory before realizing the actual bug was upstream — a config file that looked
correct but was never being loaded, because of how the script was invoked. It's a reminder to
verify a fix's mechanism is actually active before trusting that it should have worked."

---

## (add entries here as the project progresses)
