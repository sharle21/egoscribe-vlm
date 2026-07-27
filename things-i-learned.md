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

## (add entries here as the project progresses)
