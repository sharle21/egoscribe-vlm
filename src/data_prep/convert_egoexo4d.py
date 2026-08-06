"""Convert Ego-Exo4D keystep annotations into the flat schema EgocentricHOIDataset expects.

Ego-Exo4D does NOT natively provide hand-object-interaction labels (tool/object/verb/
state-change/safety-gear). Its `keystep` annotations give procedural step segments
(start_time/end_time in seconds + a short step_name/step_description), and its
`atomic_descriptions` give single-timestamp narrations. Neither is a direct match for
`src/schema.py::HandObjectInteraction`.

This script does the mechanical, trustworthy part (locating the egocentric video per take,
converting seconds -> frame indices). For labels, it prefers an LLM-extracted cache (see
src/data_prep/llm_label_segments.py) via --llm_labels_cache, and falls back to a HEURISTIC,
NOT-TRUSTWORTHY naive step_name split for any segment the cache doesn't cover. Per ADR-0004,
a human curation/review pass over a sample of the output is still required before treating
any of it as ground truth for training or eval — the LLM pass is a large quality improvement
over the heuristic, not a guarantee.

Expected input layout (Ego-Exo4D downloader defaults):
  <root>/takes.json
  <root>/annotations/keystep_train.json  (or keystep_val.json / keystep_test.json)
  <root>/takes/<take_name>/frame_aligned_videos/<cam_id>/<stream_id>.mp4

Usage:
  python -m src.data_prep.convert_egoexo4d \
      --takes_json /path/to/takes.json \
      --keystep_json /path/to/annotations/keystep_train.json \
      --video_root /path/to/takes \
      --output data/converted/annotations.json \
      [--scenario "Cooking"] [--limit 500]
"""
import argparse
import json
from pathlib import Path

from decord import VideoReader, cpu


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_egocentric_relative_path(take: dict) -> str | None:
    """Ego-Exo4D convention: camera ids starting with 'aria' are egocentric; the 'rgb'
    stream is the standard color video. Returns the relative path to the mp4, or None if
    this take has no aria/rgb stream (some takes may be exo-only or use a different stream
    layout — skip those rather than guessing).
    """
    frame_aligned_videos = take.get("frame_aligned_videos", {})
    for cam_id, streams in frame_aligned_videos.items():
        if not cam_id.lower().startswith("aria"):
            continue
        rgb_stream = streams.get("rgb")
        if rgb_stream and rgb_stream.get("relative_path"):
            return rgb_stream["relative_path"]
    return None


def heuristic_label_from_keystep(segment: dict) -> tuple[dict, dict]:
    """First-pass, NOT ground truth. Naive split of step_name into a verb-ish first token
    and an object-ish remainder. Real tool/safety-gear/state labels aren't derivable from
    keystep text alone — left as unknown defaults.

    Returns (expected_output, review_meta) as two separate dicts. `expected_output` must
    stay a clean match for `HandObjectInteraction` since `EgocentricHOIDataset` serializes
    it verbatim as the training label text — mixing review/debug fields into it would train
    the model to produce the wrong JSON shape. `review_meta` carries provenance for the
    human curation pass (ADR-0004) and is stored alongside, not inside, expected_output.
    """
    step_name = segment.get("step_name", "").strip()
    words = step_name.split(" ", 1)
    action_verb = words[0].lower() if words else ""
    target_object = words[1] if len(words) > 1 else step_name

    expected_output = {
        "tool_detected": None,  # not derivable from keystep text; needs manual/model-assisted pass
        "target_object": target_object,
        "action_verb": action_verb,
        # procedural_mistake only exists in the val/test keystep files, not train (per
        # Ego-Exo4D docs) — treat its absence as "unknown", not "False".
        "point_of_no_return_detected": bool(segment.get("procedural_mistake", False)),
        "current_state": segment.get("step_description", step_name),
        "safety_gear_missing": [],  # not derivable from keystep text; needs manual/model-assisted pass
    }
    review_meta = {
        "needs_review": True,
        "source_step_id": segment.get("step_id"),
        "source_step_unique_id": segment.get("step_unique_id"),
    }
    return expected_output, review_meta


def label_from_llm_cache(segment: dict, take_cache: dict) -> tuple[dict, dict] | None:
    """Returns (expected_output, review_meta) if this segment's step_unique_id is present in
    the take's LLM-extracted cache, else None (caller falls back to the heuristic).
    """
    key = str(segment.get("step_unique_id"))
    if take_cache is None or key not in take_cache:
        return None
    expected_output = take_cache[key]
    review_meta = {
        "needs_review": False,  # LLM-extracted, not naive heuristic — still spot-check per ADR-0004
        "source": "llm",
        "source_step_id": segment.get("step_id"),
        "source_step_unique_id": segment.get("step_unique_id"),
    }
    return expected_output, review_meta


def seconds_to_frame(seconds: float, fps: float, total_frames: int) -> int:
    frame = round(seconds * fps)
    return max(0, min(frame, total_frames - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--takes_json", required=True)
    parser.add_argument("--keystep_json", required=True)
    parser.add_argument("--video_root", required=True, help="Directory containing per-take video folders")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default=None, help="Only convert takes with this scenario name")
    parser.add_argument("--limit", type=int, default=None, help="Cap number of segments (for a curated subset, ADR-0004)")
    parser.add_argument(
        "--llm_labels_cache", default=None,
        help="Path to output of src/data_prep/llm_label_segments.py. If given, segments found "
             "in the cache use the LLM-extracted labels; segments not found fall back to the "
             "naive heuristic split (with needs_review=True) same as when this flag is omitted.",
    )
    args = parser.parse_args()

    takes = {t["take_uid"]: t for t in load_json(args.takes_json)}
    keystep = load_json(args.keystep_json)
    annotations_by_take = keystep.get("annotations", {})
    llm_cache = load_json(args.llm_labels_cache) if args.llm_labels_cache else {}

    video_root = Path(args.video_root)
    output_records = []
    stats = {
        "takes_seen": 0,
        "takes_not_in_takes_json": 0,  # has keystep annotations but missing from takes.json
        "takes_wrong_scenario": 0,
        "takes_missing_ego_video": 0,
        "takes_missing_video_file": 0,
        "segments_converted": 0,
        "segments_skipped_bad_video": 0,
        "segments_from_llm_cache": 0,
        "segments_from_heuristic": 0,
    }

    for take_uid, take_anno in annotations_by_take.items():
        stats["takes_seen"] += 1
        take = takes.get(take_uid)
        if take is None:
            stats["takes_not_in_takes_json"] += 1
            continue
        if args.scenario and take.get("scenario") != args.scenario:
            stats["takes_wrong_scenario"] += 1
            continue

        rel_path = find_egocentric_relative_path(take)
        if rel_path is None:
            stats["takes_missing_ego_video"] += 1
            continue

        video_path = video_root / take.get("take_name", take_uid) / "frame_aligned_videos" / rel_path
        if not video_path.exists():
            stats["takes_missing_video_file"] += 1
            continue

        try:
            vr = VideoReader(str(video_path), ctx=cpu(0))
            fps = vr.get_avg_fps()
            total_frames = len(vr)
        except Exception:
            stats["segments_skipped_bad_video"] += 1
            continue

        take_llm_cache = llm_cache.get(take_uid)
        for segment in take_anno.get("segments", []):
            start_frame = seconds_to_frame(segment["start_time"], fps, total_frames)
            end_frame = seconds_to_frame(segment["end_time"], fps, total_frames)
            if end_frame <= start_frame:
                continue

            llm_result = label_from_llm_cache(segment, take_llm_cache)
            if llm_result is not None:
                expected_output, review_meta = llm_result
                stats["segments_from_llm_cache"] += 1
            else:
                expected_output, review_meta = heuristic_label_from_keystep(segment)
                stats["segments_from_heuristic"] += 1
            output_records.append({
                "video_file": str(video_path.relative_to(video_root)),
                "interaction_start_frame": start_frame,
                "interaction_end_frame": end_frame,
                "expected_output": expected_output,
                "review_meta": review_meta,
            })
            stats["segments_converted"] += 1

            if args.limit and len(output_records) >= args.limit:
                break
        if args.limit and len(output_records) >= args.limit:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_records, f, indent=2)

    print(f"Wrote {len(output_records)} records to {args.output}")
    print(f"Stats: {stats}")
    if args.llm_labels_cache:
        print(
            f"{stats['segments_from_llm_cache']} segments used LLM-extracted labels, "
            f"{stats['segments_from_heuristic']} fell back to the naive heuristic (not in cache). "
            "Spot-check a sample of BOTH before trusting as ground truth (ADR-0004)."
        )
    else:
        print(
            "NOTE: no --llm_labels_cache given, all expected_output entries are heuristic "
            "(needs_review=True) — tool_detected/safety_gear_missing are placeholders and "
            "action_verb/target_object are a naive split of step_name. Review a sample before "
            "trusting these as labels (ADR-0004)."
        )


if __name__ == "__main__":
    main()
