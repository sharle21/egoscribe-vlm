"""Run one adapter-backed EgoScribe inference example.

This is the checkpoint-serving sanity path. It deliberately uses the same Unsloth + PEFT
loader, prompt, frame sampling, and schema validation as ``src.eval.evaluate``. The old
SGLang demo loaded only the untouched base model and therefore could not test a trained
strategy adapter.

Example:
    python serve.py \
        --annotations data/converted/annotations_eval.json \
        --video_dir data/ego-exo4d/takes \
        --adapter saved_egoscribe_adapters/strategy_C \
        --index 0
"""

import argparse
import json
from types import SimpleNamespace

from src.eval.evaluate import extract_json, generate_one, load_model
from src.schema import HandObjectInteraction


def parse_args():
    parser = argparse.ArgumentParser(description="Run one EgoScribe adapter inference.")
    parser.add_argument("--annotations", required=True, help="Annotation JSON containing the example.")
    parser.add_argument("--video_dir", required=True, help="Root directory for video_file paths.")
    parser.add_argument("--adapter", default=None, help="Saved LoRA adapter; omit for the base model.")
    parser.add_argument("--index", type=int, default=0, help="Annotation index to run.")
    parser.add_argument("--num_frames", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    parser.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.annotations) as f:
        records = json.load(f)
    if not 0 <= args.index < len(records):
        raise IndexError(f"--index must be between 0 and {len(records) - 1}")

    item = records[args.index]
    model, processor = load_model(args.adapter)
    generation_args = SimpleNamespace(
        num_frames=args.num_frames,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    raw = generate_one(model, processor, item, args.video_dir, generation_args)
    parsed = extract_json(raw)
    valid = False
    if parsed is not None:
        try:
            HandObjectInteraction(**parsed)
            valid = True
        except Exception:
            pass

    print(f"video: {item['video_file']}")
    print(f"adapter: {args.adapter or 'base model'}")
    print(f"schema_valid: {valid}")
    print("raw_generation:")
    print(raw)
    if parsed is not None:
        print("parsed:")
        print(json.dumps(parsed, indent=2))
    if not valid:
        raise SystemExit("Generated output did not validate against HandObjectInteraction")


if __name__ == "__main__":
    main()
