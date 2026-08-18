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
Tool detection is a secondary reliability signal; the three primary semantic fields are
target-object, action-verb, and current-state.

| Strategy | Adaptation | Params | Valid JSON | Tool F1 | Target F1 | Action F1 | State F1 | PONR F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Vision + language, attn + MLP | 51.5M (0.62%) | 110/110 | 0.773 [0.69, 0.87] | 0.313 [0.21, 0.41] | 0.242 [0.14, 0.33] | 0.182 [0.13, 0.25] | 0.429 [0.00, 0.67] |
| B | Vision only, attn + MLP | 11.2M (0.13%) | 110/110 | 0.742 [0.65, 0.85] | 0.227 [0.14, 0.36] | 0.105 [0.06, 0.16] | 0.085 [0.03, 0.18] | 0.364 [0.00, 0.67] |
| C | Language only, attn + MLP | 40.4M (0.48%) | 110/110 | 0.741 [0.65, 0.86] | 0.323 [0.22, 0.41] | 0.251 [0.17, 0.34] | 0.191 [0.14, 0.25] | 0.462 [0.00, 0.62] |
| D | Vision + language, attn only | 14.0M (0.17%) | 101/110 | 0.710 [0.58, 0.84] | 0.284 [0.20, 0.38] | 0.186 [0.10, 0.27] | 0.125 [0.09, 0.19] | 0.308 [0.00, 0.67] |

D's semantic scores are computed over its 101 schema-valid predictions; its nine invalid outputs
are counted only in the validity rate, never silently treated as correct.

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
  separation the study produces.
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
A and C is not resolvable at this sample size and is not claimed.

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
