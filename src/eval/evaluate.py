"""EgoScribe eval harness (PRD section 8).

Generates predictions on the held-out eval split with a trained strategy adapter (or the untuned
base model as a baseline) and scores them against the schema fields. Uses the SAME prompt and
frame-sampling as training (src/dataset.py) so eval isn't measuring a distribution shift; uses
the Unsloth + PEFT stack (not serve.py's SGLang path) so adapter loading is a one-liner and
matches exactly what train.py saved.

Baseline (untuned base model):
  python -m src.eval.evaluate \
      --annotations data/converted/annotations_eval.json \
      --video_dir data/ego-exo4d/takes \
      --out eval_out/baseline.json

A trained strategy:
  python -m src.eval.evaluate \
      --annotations data/converted/annotations_eval.json \
      --video_dir data/ego-exo4d/takes \
      --adapter saved_egoscribe_adapters/strategy_A \
      --out eval_out/strategy_A.json

Writes {--out}: the metrics + every per-record prediction/gold pair (for manual review of the
weak free-text labels — see metrics.py).
"""
import argparse
import json
import os
import re
from pathlib import Path

import torch
from decord import VideoReader, cpu
from unsloth import FastVisionModel

from src.dataset import build_user_messages, sample_frame_indices
from src.schema import HandObjectInteraction
from src.eval.metrics import compute_metrics

MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--annotations", required=True, help="Held-out eval JSON (annotations_eval.json).")
    p.add_argument("--video_dir", required=True, help="Root the video_file paths resolve against.")
    p.add_argument("--adapter", default=None, help="Path to a trained strategy adapter dir. Omit for untuned baseline.")
    p.add_argument("--out", required=True, help="Where to write metrics + per-record predictions JSON.")
    p.add_argument("--num_frames", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--min_pixels", type=int, default=64 * 28 * 28)
    p.add_argument("--max_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--limit", type=int, default=-1, help="Only eval first N records (smoke test). -1 = all.")
    return p.parse_args()


def load_model(adapter):
    """Base 4-bit model, optionally with a trained adapter merged in for inference."""
    model, processor = FastVisionModel.from_pretrained(MODEL_ID, load_in_4bit=True)
    if adapter:
        # Adapters saved by train.py via peft save_pretrained; load them back onto the base.
        model.load_adapter(adapter)
    FastVisionModel.for_inference(model)
    return model, processor


def extract_json(text):
    """Pull the first balanced {...} block out of a generation and parse it. None on failure."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None


def generate_one(model, processor, item, video_dir, args):
    video_path = os.path.join(video_dir, item["video_file"])
    vr = VideoReader(video_path, ctx=cpu(0))
    idx = sample_frame_indices(item["interaction_start_frame"], item["interaction_end_frame"],
                               len(vr), args.num_frames)
    frames = list(vr.get_batch(idx).asnumpy())

    messages = build_user_messages(len(frames))
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=frames, min_pixels=args.min_pixels,
                       max_pixels=args.max_pixels, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
    # Decode only the newly generated tokens, not the echoed prompt.
    gen = out[0][inputs["input_ids"].shape[1]:]
    return processor.tokenizer.decode(gen, skip_special_tokens=True)


def main():
    args = parse_args()
    records = json.load(open(args.annotations))
    if args.limit > 0:
        records = records[:args.limit]

    model, processor = load_model(args.adapter)

    preds, golds, per_record = [], [], []
    for i, item in enumerate(records):
        raw = generate_one(model, processor, item, args.video_dir, args)
        parsed = extract_json(raw)
        valid = False
        if parsed is not None:
            try:
                HandObjectInteraction(**parsed)  # schema validation
                valid = True
            except Exception:
                valid = False
        if valid:
            preds.append(parsed)
            golds.append(item["expected_output"])
        per_record.append({
            "video_file": item["video_file"],
            "gold": item["expected_output"],
            "raw_generation": raw,
            "parsed": parsed,
            "schema_valid": valid,
        })
        print(f"[{i + 1}/{len(records)}] {item['video_file']} valid={valid}")

    metrics = compute_metrics(preds, golds, num_total=len(records))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "config": {"adapter": args.adapter, "annotations": args.annotations,
                       "num_frames": args.num_frames, "model_id": MODEL_ID},
            "metrics": metrics,
            "predictions": per_record,
        }, f, indent=2)

    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
