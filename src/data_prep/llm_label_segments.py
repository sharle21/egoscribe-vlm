"""LLM-assisted labeling for Ego-Exo4D keystep segments, replacing the naive heuristic split
in convert_egoexo4d.py::heuristic_label_from_keystep.

Why: heuristic step_name splitting (verb = first word, object = rest) breaks on compound/
clause-heavy step names (~half of real segments — see things-i-learned.md, 2026-08 entries)
and can never populate tool_detected/safety_gear_missing since that info simply isn't in
keystep text. atomic_descriptions narrations ARE per-timestamp and DO mention hands/tools
concretely (e.g. "C touches the screw of the hub with his right hand"), so this script
cross-references each keystep segment's [start_time, end_time] window against the take's
atomic_descriptions to give the LLM real material to extract from, then uses Claude's
structured tool-calling to force output into the exact HandObjectInteraction schema.

Output is a lookup cache — {take_uid: {step_unique_id: HandObjectInteraction dict}} — that
convert_egoexo4d.py can consume via --llm_labels_cache, falling back to the heuristic (with
needs_review=True) for any segment not present in the cache.

Usage:
  export ANTHROPIC_API_KEY=...   # set in your own shell, never pass as a CLI arg/commit it
  python -m src.data_prep.llm_label_segments \
      --keystep_json /path/to/keystep_train.json \
      --atomic_json /path/to/atomic_descriptions_train.json \
      --output data/converted/llm_labels_cache.json \
      [--scenario "Covid-19 Rapid Antigen Test"] [--limit 500]

Cost/model: defaults to Claude Haiku 4.5 — cheap/fast, appropriate for a structured-extraction
task (not open-ended reasoning) across thousands of segments. Override with --model if a
spot-check shows Haiku's outputs too noisy for a particular scenario.
"""
import argparse
import json
import time
from pathlib import Path

import anthropic

from src.schema import HandObjectInteraction

MODEL = "claude-haiku-4-5-20251001"

EXTRACTION_TOOL = {
    "name": "extract_hand_object_interaction",
    "description": "Extract structured hand-object-interaction fields from an egocentric procedural step.",
    "input_schema": {
        "type": "object",
        "properties": {
            "tool_detected": {
                "type": ["string", "null"],
                "description": "The tool being held/used, e.g. 'screwdriver'. Null if no tool is used or none is mentioned.",
            },
            "target_object": {
                "type": "string",
                "description": "The main object being manipulated, e.g. 'circuit_board'. Just the object noun phrase, not the full sentence.",
            },
            "action_verb": {
                "type": "string",
                "description": "The precise action verb, e.g. 'unscrewing', 'soldering'. Single verb or verb phrase, not a full sentence.",
            },
            "point_of_no_return_detected": {
                "type": "boolean",
                "description": "True only if this step describes an irreversible physical state change (e.g. cutting, breaking a seal, permanently attaching). False for reversible/inspection actions.",
            },
            "current_state": {
                "type": "string",
                "description": "The resulting structural state after this step, e.g. 'disassembled', 'secured', 'open'.",
            },
            "safety_gear_missing": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Safety gear that should be worn but is visibly absent, e.g. ['gloves']. Empty list if not mentioned/not applicable — do not guess.",
            },
        },
        "required": ["target_object", "action_verb", "point_of_no_return_detected", "current_state", "safety_gear_missing"],
    },
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def gather_narrations_in_window(atomic_records, start_time, end_time):
    """atomic_records: list of annotation dicts for one take (may include rejected ones).
    Uses the first non-rejected annotator's descriptions, matching the convention docs
    describe (one take can have multiple annotators; we don't need all of them for context).
    """
    for record in atomic_records:
        if record.get("rejected"):
            continue
        texts = [
            d["text"] for d in record.get("descriptions", [])
            if start_time <= d.get("timestamp", -1) <= end_time
        ]
        if texts:
            return texts
    return []


def build_prompt(step_name, step_description, narrations):
    narration_block = "\n".join(f"- {t}" for t in narrations) if narrations else "(none available)"
    return (
        f"Procedural step name: {step_name!r}\n"
        f"Step description: {step_description!r}\n"
        f"Egocentric narrations during this step's time window:\n{narration_block}\n\n"
        "Extract the structured hand-object-interaction fields for this step using the "
        "extract_hand_object_interaction tool. Base tool_detected and safety_gear_missing "
        "strictly on what's mentioned in the narrations/description above — do not guess or "
        "infer objects/tools that aren't actually referenced."
    )


# Haiku 4.5 pricing, $/1M tokens — used only to report an estimated cost at the end of a run,
# not sent to the API. Update if pricing changes or --model is overridden to something else.
HAIKU_INPUT_PER_MTOK = 1.00
HAIKU_OUTPUT_PER_MTOK = 5.00


def extract_one(client, model, step_name, step_description, narrations, max_retries=3):
    prompt = build_prompt(step_name, step_description, narrations)
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=512,
                tools=[EXTRACTION_TOOL],
                tool_choice={"type": "tool", "name": "extract_hand_object_interaction"},
                messages=[{"role": "user", "content": prompt}],
            )
            for block in response.content:
                if block.type == "tool_use":
                    # Validate through the actual schema, not just trust the tool call —
                    # normalizes optional fields (e.g. tool_detected) to an explicit null when
                    # Claude omits the key rather than returning null, so every cached record
                    # has the same JSON shape as every other (matters since dataset.py trains
                    # on json.dumps(expected_output) verbatim — inconsistent shape across
                    # examples would give the model a noisier target to learn).
                    result = HandObjectInteraction(**block.input).model_dump()
                    return result, response.usage.input_tokens, response.usage.output_tokens
            raise ValueError("No tool_use block in response")
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keystep_json", required=True)
    parser.add_argument("--atomic_json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--limit", type=int, default=None, help="Cap number of segments (cost control)")
    parser.add_argument("--model", default=MODEL)
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

    keystep = load_json(args.keystep_json)
    atomic = load_json(args.atomic_json)
    keystep_annos = keystep.get("annotations", {})
    atomic_annos = atomic.get("annotations", {})

    cache = {}
    n_done = 0
    n_failed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    for take_uid, take_anno in keystep_annos.items():
        if args.scenario and take_anno.get("scenario") != args.scenario:
            continue

        take_atomic_records = atomic_annos.get(take_uid, [])
        take_cache = {}

        for segment in take_anno.get("segments", []):
            if args.limit and n_done >= args.limit:
                break

            narrations = gather_narrations_in_window(
                take_atomic_records, segment["start_time"], segment["end_time"]
            )
            try:
                extracted, in_tok, out_tok = extract_one(
                    client, args.model,
                    segment["step_name"], segment["step_description"], narrations,
                )
                take_cache[str(segment["step_unique_id"])] = extracted
                total_input_tokens += in_tok
                total_output_tokens += out_tok
                n_done += 1
            except Exception as e:
                n_failed += 1
                print(f"FAILED take={take_uid} step_unique_id={segment.get('step_unique_id')}: {e}")

        if take_cache:
            cache[take_uid] = take_cache

        if args.limit and n_done >= args.limit:
            break

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cache, f, indent=2)

    print(f"Wrote {n_done} labeled segments ({n_failed} failed) across {len(cache)} takes to {args.output}")
    if args.model == MODEL:
        cost = (total_input_tokens / 1e6) * HAIKU_INPUT_PER_MTOK + (total_output_tokens / 1e6) * HAIKU_OUTPUT_PER_MTOK
        print(
            f"Tokens: {total_input_tokens} in / {total_output_tokens} out "
            f"— est. cost ${cost:.4f} at Haiku 4.5 rates (${HAIKU_INPUT_PER_MTOK}/${HAIKU_OUTPUT_PER_MTOK} per 1M in/out)"
        )
    else:
        print(f"Tokens: {total_input_tokens} in / {total_output_tokens} out (cost not estimated — non-default --model)")


if __name__ == "__main__":
    main()
