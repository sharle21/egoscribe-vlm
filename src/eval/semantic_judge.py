"""Semantic (LLM-judge) rescoring of the free-text fields, offline from saved prediction dumps.

Why this exists
---------------
The headline free-text metric is SQuAD-style token-F1 (src/eval/metrics.py). Token-F1 measures
*lexical* overlap, which is the right proxy only when an answer has a small canonical surface
form. These fields are generative and paraphrastic: gold "beating the eggs" vs pred "whisking"
is correct but shares zero tokens, and verbose gold ("covid-19 test box and its contents (test
tube pack, ...)") vs a clean pred ("COVID-19 test kit box") is penalized for being concise. So
token-F1 systematically *under*-measures true agreement here.

This module adds a semantic score using Claude Haiku 4.5 as a judge. Crucially it judges the
SAME quantity token-F1 does — agreement between the prediction and the gold LABEL — not
correctness against the video. The judge sees only (field, gold, pred) text, so this is an
apples-to-apples upgrade of *how* agreement is measured (meaning vs surface form), not a change
of *what* is measured. Report both: token-F1 is the cheap deterministic floor, the semantic
score is the fair measure, and the gap between them quantifies how much of the low token-F1 was
metric harshness rather than model error.

Runs entirely on the saved docs/eval_out/strategy_*_clustered.json dumps — no GPU, no retraining.
Judgments are cached by (field, gold, pred) so re-runs and repeated pairs are free.

Usage:
    python -m src.eval.semantic_judge docs/eval_out/strategy_A_clustered.json \
        docs/eval_out/strategy_B_clustered.json ...
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import anthropic

from src.eval.metrics import _percentile  # reuse the exact percentile used for token-F1 CIs

MODEL = "claude-haiku-4-5-20251001"
FREE_TEXT_FIELDS = ["target_object", "action_verb", "current_state"]  # tool_detected is mostly null
CACHE_PATH = Path("docs/eval_out/semantic_judge_cache.json")

# Haiku 4.5 pricing $/1M tokens, for the end-of-run cost estimate only.
HAIKU_INPUT_PER_MTOK = 1.00
HAIKU_OUTPUT_PER_MTOK = 5.00

_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}

JUDGE_TOOL = {
    "name": "score_fields",
    "description": "Rate how well each predicted field agrees IN MEANING with the gold label.",
    "input_schema": {
        "type": "object",
        "properties": {
            f: {
                "type": "string",
                "enum": ["correct", "partial", "wrong"],
                "description": (
                    "correct = same meaning as gold (synonym/paraphrase/more-or-less specific "
                    "but clearly the same thing); partial = related and overlapping but missing "
                    "or adding a meaningful distinction; wrong = different thing."
                ),
            }
            for f in FREE_TEXT_FIELDS
        },
        "required": FREE_TEXT_FIELDS,
    },
}


def take_of(video_file):
    """Take id = first path component (e.g. 'georgiatech_covid_12_3/frame_aligned_videos/...')."""
    return video_file.split("/", 1)[0]


def build_prompt(gold, pred):
    lines = ["Judge whether each predicted value means the SAME as the gold value for an",
             "egocentric hand-object-interaction segment. Judge meaning, not wording.\n"]
    for f in FREE_TEXT_FIELDS:
        lines.append(f"{f}:\n  gold: {gold.get(f)!r}\n  pred: {pred.get(f)!r}")
    return "\n".join(lines)


def judge_one(client, model, gold, pred, cache):
    """Return {field: score} for one record, using/refreshing the (field,gold,pred) cache."""
    scores, need = {}, False
    for f in FREE_TEXT_FIELDS:
        key = f"{f}||{str(gold.get(f)).lower()}||{str(pred.get(f)).lower()}"
        if key in cache:
            scores[f] = cache[key]
        else:
            need = True
    if not need:
        return scores, 0, 0

    prompt = build_prompt(gold, pred)
    for attempt in range(3):
        try:
            resp = client.messages.create(
                model=model, max_tokens=256, temperature=0.0,
                tools=[JUDGE_TOOL],
                tool_choice={"type": "tool", "name": "score_fields"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in resp.content:
                if block.type == "tool_use":
                    for f in FREE_TEXT_FIELDS:
                        label = block.input.get(f, "wrong")
                        val = _SCORE.get(label, 0.0)
                        scores[f] = val
                        cache[f"{f}||{str(gold.get(f)).lower()}||{str(pred.get(f)).lower()}"] = val
                    return scores, resp.usage.input_tokens, resp.usage.output_tokens
            raise ValueError("no tool_use block")
        except (anthropic.RateLimitError, anthropic.APIStatusError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)


def bootstrap_semantic(per_record, take_ids, n_boot=1000, seed=0):
    """Take-level block bootstrap over semantic scores — same design as metrics.take_level_bootstrap."""
    import random
    idx_by_take = defaultdict(list)
    for i, t in enumerate(take_ids):
        idx_by_take[t].append(i)
    takes = list(idx_by_take)
    if len(takes) < 2:
        return {}
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        idxs = []
        for _ in takes:
            idxs.extend(idx_by_take[rng.choice(takes)])
        for f in FREE_TEXT_FIELDS:
            vals = [per_record[i][f] for i in idxs]
            samples[f].append(sum(vals) / len(vals))
    ci = {}
    for f, vals in samples.items():
        vals.sort()
        ci[f] = [round(_percentile(vals, 2.5), 4), round(_percentile(vals, 97.5), 4)]
    return ci


def rescore(path, client, model, cache):
    data = json.load(open(path))
    preds = [p for p in data["predictions"] if p.get("parsed")]
    per_record, take_ids = [], []
    in_tok = out_tok = 0
    for i, p in enumerate(preds):
        s, it, ot = judge_one(client, model, p["gold"], p["parsed"], cache)
        per_record.append(s)
        take_ids.append(take_of(p["video_file"]))
        in_tok += it
        out_tok += ot
        if (i + 1) % 20 == 0:
            print(f"  {path.split('/')[-1]}: judged {i + 1}/{len(preds)}")
    means = {f: round(sum(r[f] for r in per_record) / len(per_record), 3) for f in FREE_TEXT_FIELDS}
    ci = bootstrap_semantic(per_record, take_ids)
    return means, ci, in_tok, out_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dumps", nargs="+", help="strategy_*_clustered.json prediction dumps")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    cache = json.load(open(CACHE_PATH)) if CACHE_PATH.exists() else {}
    client = anthropic.Anthropic()
    total_in = total_out = 0
    print(f"{'strategy':<12} " + " ".join(f"{f:>16}" for f in FREE_TEXT_FIELDS))
    for path in args.dumps:
        means, ci, it, ot = rescore(path, client, args.model, cache)
        total_in += it
        total_out += ot
        CACHE_PATH.write_text(json.dumps(cache, indent=2))  # persist incrementally
        name = Path(path).stem.replace("_clustered", "")
        cells = []
        for f in FREE_TEXT_FIELDS:
            lo, hi = ci.get(f, [None, None])
            span = f"[{lo:.2f},{hi:.2f}]" if lo is not None else "[--,--]"
            cells.append(f"{means[f]:.3f} {span:<13}")
        out_path = path.replace("_clustered.json", "_semantic.json")
        json.dump({"semantic_mean": means, "semantic_ci95": ci, "judge_model": args.model},
                  open(out_path, "w"), indent=2)
        print(f"{name:<12} " + " ".join(cells))
    cost = total_in / 1e6 * HAIKU_INPUT_PER_MTOK + total_out / 1e6 * HAIKU_OUTPUT_PER_MTOK
    print(f"\nAPI: {total_in} in / {total_out} out tokens ≈ ${cost:.2f} "
          f"(cached pairs reused, so re-runs are cheaper).")


if __name__ == "__main__":
    main()
