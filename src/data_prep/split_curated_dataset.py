"""Build the held-out train/eval split for the curated Ego-Exo4D subset (ADR-0004).

Per ADR-0004, the split happens ONCE, by take_uid (not by segment) — splitting by segment
would let frames from the same take appear in both train and eval, leaking context (lighting,
camera angle, the specific person's hands) and inflating eval scores. The resulting two files
are reused identically across all 4 strategy runs (A/B/C/D) so comparisons stay apples-to-apples
(ADR-0002's requirement that only the adaptation config differs between runs).

Split is stratified by scenario (domain) with a fixed seed, so eval isn't accidentally all one
domain — each of the 3 curated scenarios contributes its own held-out takes.

Usage:
  python -m src.data_prep.split_curated_dataset \
      --annotations data/converted/annotations.json \
      --takes_json /path/to/takes.json \
      --keystep_json /path/to/annotations/keystep_train.json \
      --eval_fraction 0.2 \
      --seed 42 \
      --train_output data/converted/annotations_train.json \
      --eval_output data/converted/annotations_eval.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True, help="Output of convert_egoexo4d.py")
    parser.add_argument("--takes_json", required=True)
    parser.add_argument("--keystep_json", required=True)
    parser.add_argument("--eval_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--eval_output", required=True)
    args = parser.parse_args()

    records = load_json(args.annotations)
    takes_by_name = {t["take_name"]: t["take_uid"] for t in load_json(args.takes_json)}
    keystep_annos = load_json(args.keystep_json)["annotations"]

    # Group records by take_name (first path component of video_file), then resolve each
    # take's scenario so the split can be stratified per domain.
    records_by_take = defaultdict(list)
    for rec in records:
        take_name = rec["video_file"].split("/", 1)[0]
        records_by_take[take_name].append(rec)

    takes_by_scenario = defaultdict(list)
    for take_name in records_by_take:
        take_uid = takes_by_name.get(take_name)
        scenario = keystep_annos.get(take_uid, {}).get("scenario", "UNKNOWN") if take_uid else "UNKNOWN"
        takes_by_scenario[scenario].append(take_name)

    rng = random.Random(args.seed)
    train_records, eval_records = [], []
    split_summary = {}

    for scenario, take_names in takes_by_scenario.items():
        take_names = sorted(take_names)  # deterministic order before shuffling
        rng.shuffle(take_names)
        n_eval = max(1, round(len(take_names) * args.eval_fraction))
        eval_takes = set(take_names[:n_eval])
        train_takes = set(take_names[n_eval:])

        split_summary[scenario] = {"train_takes": len(train_takes), "eval_takes": len(eval_takes)}

        for take_name in train_takes:
            train_records.extend(records_by_take[take_name])
        for take_name in eval_takes:
            eval_records.extend(records_by_take[take_name])

    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.train_output, "w") as f:
        json.dump(train_records, f, indent=2)
    with open(args.eval_output, "w") as f:
        json.dump(eval_records, f, indent=2)

    print(f"Train: {len(train_records)} segments across {sum(s['train_takes'] for s in split_summary.values())} takes")
    print(f"Eval:  {len(eval_records)} segments across {sum(s['eval_takes'] for s in split_summary.values())} takes")
    print(f"Per-scenario split: {json.dumps(split_summary, indent=2)}")
    print(f"Wrote {args.train_output} and {args.eval_output}")


if __name__ == "__main__":
    main()
