# EgoScribe results summary

## Experimental setup

- Model: Qwen2.5-VL-7B-Instruct with 4-bit QLoRA via Unsloth.
- Training data: 572 segments from 24 takes.
- Held-out evaluation: 110 segments from 6 disjoint takes.
- Each strategy used the same split, seed, and three-epoch training recipe.
- Confidence intervals are 95% block-bootstrap intervals over the six takes (1000 resamples),
  not over segments. Takes are the independent statistical unit.
- Source data: the committed `docs/eval_out/strategy_{A,B,C,D}_clustered.json` files. Every
  number below is reproducible from those files.

## Strategy comparison

Values are token-F1 point estimates with the 95% take-level bootstrap interval in brackets.
Token-F1 is the strict *lexical* floor — a fairer semantic rescore of the free-text fields
follows in the next section. Tool detection is a secondary reliability signal; the three primary
semantic fields are target-object, action-verb, and current-state.

| Strategy | Adaptation | Params | Valid JSON | Tool F1 | Target F1 | Action F1 | State F1 | PONR F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Vision + language, attn + MLP | 51.5M (0.62%) | 110/110 | 0.773 [0.69, 0.87] | 0.313 [0.21, 0.41] | 0.242 [0.14, 0.33] | 0.182 [0.13, 0.25] | 0.429 [0.00, 0.67] |
| B | Vision only, attn + MLP | 11.2M (0.13%) | 110/110 | 0.742 [0.65, 0.85] | 0.227 [0.14, 0.36] | 0.105 [0.06, 0.16] | 0.085 [0.03, 0.18] | 0.364 [0.00, 0.67] |
| C | Language only, attn + MLP | 40.4M (0.48%) | 110/110 | 0.741 [0.65, 0.86] | 0.323 [0.22, 0.41] | 0.251 [0.17, 0.34] | 0.191 [0.14, 0.25] | 0.462 [0.00, 0.62] |
| D | Vision + language, attn only | 14.0M (0.17%) | 101/110 | 0.710 [0.58, 0.84] | 0.284 [0.20, 0.38] | 0.186 [0.10, 0.27] | 0.125 [0.09, 0.19] | 0.308 [0.00, 0.67] |

D's scores are computed over its 101 schema-valid predictions; its nine invalid outputs are
counted only in the validity rate, never silently treated as correct.

## Semantic (LLM-judge) rescore of the free-text fields

Token-F1 measures *lexical* overlap. These fields are generative and paraphrastic — gold
"beating the eggs" vs pred "whisking" is correct but shares zero tokens, and a concise pred is
penalized against verbose LLM-written gold. So token-F1 systematically *under*-measures true
agreement. To separate "bad model" from "harsh metric", each free-text field was rescored with a
Claude Haiku 4.5 judge that rates prediction-vs-gold agreement in *meaning* (correct=1.0 /
partial=0.5 / wrong=0.0). The judge sees only the gold and predicted text, so it measures the
same quantity token-F1 does (agreement with the gold label), just semantically. Offline from the
saved dumps, no retraining (`src/eval/semantic_judge.py`, ~$0.42).

Semantic mean with 95% take-level bootstrap interval:

| Strategy | Target | Action | State |
|---|---|---|---|
| A | 0.336 [0.25, 0.44] | 0.364 [0.22, 0.50] | 0.405 [0.32, 0.51] |
| B | 0.305 [0.23, 0.39] | 0.291 [0.17, 0.40] | 0.282 [0.21, 0.38] |
| C | 0.350 [0.24, 0.47] | 0.345 [0.22, 0.47] | 0.391 [0.30, 0.50] |
| D | 0.312 [0.23, 0.45] | 0.287 [0.14, 0.42] | 0.356 [0.28, 0.45] |

Two things this establishes:

- **The model works; token-F1 was underselling it.** The gap is largest on `current_state`
  (token-F1 ~0.18 → semantic ~0.40, ~2.2x) and `action_verb` (~0.24 → ~0.36) — exactly the
  paraphrase-heavy fields. The honest absolute level is ~0.35–0.40 semantic agreement: *modest
  but clearly functional*, around the "partial" mark on average, not the near-broken 0.18 that
  lexical scoring implied. It is **not** inflated to "strong" — the model is genuinely middling.
- **The comparison is unchanged — no manufactured winner.** A and C remain statistically
  indistinguishable (overlapping intervals on all three fields), B remains weakest, and every
  interval still overlaps. The semantic metric corrected the *absolute* picture without inventing
  a *ranking*, which is the honesty check the fix has to pass.

It also *sharpens* the main finding: `current_state` — the most language-loaded field — is both
where token-F1 undersold most and where vision-only (B, 0.282) trails A/C (~0.40) most clearly.
Adapting the language side helps *describe state*, which is precisely where a vision-only adapter
falls short.

## What the intervals actually license

The take-level intervals are wide, and this changes the honest reading of the table. Reading the
point estimates alone would overclaim. Specifically:

- **A and C are statistically indistinguishable on every field.** Their intervals overlap almost
  completely on target, action, and state. The small point-estimate lead C has (e.g. 0.251 vs
  0.242 on action) is well inside the noise. There is no evidence here that C beats A, nor that A
  beats C.
- **Vision-only (B) is the one strategy that trends clearly worst on the language-shaped fields.**
  On action-verb, B's interval [0.06, 0.16] sits almost entirely below C's [0.17, 0.34] and below
  A's [0.14, 0.33]; on current-state B is similarly low. This is the closest thing to a
  separation the study produces, and the semantic rescore above shows the same ordering (B state
  0.282 vs A/C ~0.40), so it is not a token-matching artifact.
- **Point-of-no-return F1 is uninformative.** Every interval spans roughly [0.0, 0.65] on only
  five positive examples across six takes. No PONR ranking claim is supportable.
- **Tool detection does not separate the strategies** — all four intervals overlap.

## Primary conclusion

Directionally, the data point to the domain gap living on the language/description side, not the
seeing side: adapting the **language decoder alone (C)** matches full vision+language adaptation
(A) on every measured field, at 0.48% vs 0.62% trainable params, while adapting the **vision
encoder alone (B)** is the weakest configuration on the semantic fields. D shows that dropping the
MLP (attention-only) costs both schema reliability (101/110) and semantic quality without a
compensating win.

The efficiency framing is where the finding is strongest and least noise-sensitive: C reaches
A-equal quality with a strictly smaller, language-scoped adapter. The per-field *ranking* between
A and C is not resolvable at this sample size and is not claimed. Both the strict lexical (token-F1)
and the semantic (LLM-judge) metrics agree on this picture, so the conclusion does not depend on
the choice of metric — only the absolute scores do (the semantic rescore roughly doubles
`current_state`, confirming the model is functional rather than broken).

This is an exploratory result over six held-out takes, not a statistically definitive ranking.

## Not measurable in this split

- **Safety-gear detection:** the held-out gold set contains zero positive `safety_gear_missing`
  items. Its 0.0 F1 is reported as *unavailable*, not as evidence the model solved the task.

## Cost

Total actual GCP spend for the whole study: **$16.25** (Compute Engine $16.19 + Networking
$0.06), from the billing console. Fully covered by the $300 free-trial credit, so **$0 out of
pocket**.

Of the $16.19 compute, roughly $7 is the four training runs at L4 list price (~$0.71/hr ×
~2.4–2.7h each, ~$1.7–1.9 per strategy); the remainder is VM overhead — setup, data transfer,
the four evaluation passes, benchmarking, and idle uptime. The $16 all-in is the honest project
cost; the ~$2/strategy marginal figure is what a fifth strategy would add. Training wall times:
A 2h40m, B 2h27m, C 2h26m, D 2h23m.

## Open items (not yet closed)

- **Serving path is unverified.** `serve.py` is now an adapter-backed single-example runner that
  reuses the eval loader, prompt, frame sampling, and schema validation (`--adapter <dir>
  --index <n>`), so a saved A–D checkpoint can be served. It has not yet been run end-to-end on a
  GPU. Reported as code-ready, not verified.
