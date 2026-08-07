"""Recompute metrics (incl. take-level bootstrap CIs) from an existing eval dump — no GPU.

evaluate.py writes every per-record prediction/gold into its output JSON. When the metric code
changes (e.g. take-level bootstrap was added after A/B/C had already been evaluated), there's no
need to re-run the model: this replays the stored predictions through the current
compute_metrics. Lets already-paid-for runs pick up honest CIs for free.

  python -m src.eval.rescore --in eval_out/strategy_A.json --out eval_out/strategy_A.json

Reads {--in}'s "predictions" block, rebuilds preds/golds/take_ids from the schema-valid records,
and overwrites (or writes to --out) the "metrics" block. Predictions are left untouched.
"""
import argparse
import json
from pathlib import Path

from src.eval.metrics import compute_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True, help="An eval_out JSON from evaluate.py.")
    p.add_argument("--out", default=None, help="Where to write. Defaults to --in (in place).")
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()

    data = json.load(open(args.inp))
    records = data["predictions"]

    preds, golds, take_ids = [], [], []
    for r in records:
        if r.get("schema_valid"):
            preds.append(r["parsed"])
            golds.append(r["gold"])
            take_ids.append(r["video_file"].split("/", 1)[0])

    data["metrics"] = compute_metrics(preds, golds, num_total=len(records),
                                      take_ids=take_ids, n_boot=args.n_boot)

    out = args.out or args.inp
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Rescored {len(preds)}/{len(records)} valid records from {len(set(take_ids))} takes -> {out}")
    print(json.dumps(data["metrics"], indent=2))


if __name__ == "__main__":
    main()
