# tasks.md — EgoScribe working checklist

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Phase 0 — Setup & code fixes
- [ ] `egodex_test.zip` (17GB, local) is the wrong dataset — confirm safe to delete once
      Ego-Exo4D access/download is sorted, to free disk space
- [x] Resolve model choice ([ADR-0001](adr/0001-model-selection.md)) — Qwen2.5-VL-7B-Instruct, accepted
- [ ] Update `serve.py` (SGLang path + `model_path`) from MiMo-VL to Qwen2.5-VL to match ADR-0001
- [x] Fix label masking bug: pad `labels` with `-100`, not tokenizer pad id (`src/dataset.py`)
- [x] Switch `train.py` model loading from full BF16 to 4-bit QLoRA via Unsloth (`--strategy` flag implements ADR-0002 + ADR-0003)
- [x] Add Unsloth + bitsandbytes to `requirements.txt`
- [x] Split `requirements.txt` into `requirements-train.txt` / `requirements-serve.txt` — `sglang`/`flashinfer` (serving-only) was blocking training installs with a fragile CUDA source build
- [x] Fix `data/samples/annotations.json` path mismatch (was at `data/annotations.json`, code expects it under `data/samples/`)
- [x] Delete stale `data/inspect_meta.py` (leftover from the EgoDex/Apple mistake, reads HDF5 — irrelevant now)
- [x] `pip install -r requirements-train.txt` on Colab T4, confirmed Unsloth loads `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`
- [x] Fix collator: `torch.cat` not `torch.stack` for `pixel_values`/`image_grid_thw` (Qwen2.5-VL flattens image patches across the batch, doesn't stack per-sample)
- [x] Fix prompt: use `processor.apply_chat_template` for image placeholders, not a literal `"<image>"` string (wasn't a real special token — 0 placeholder tokens ever inserted)
- [x] Fix labels: concatenate prompt+answer into one sequence with prompt masked `-100`, instead of two independently-tokenized/padded sequences of different lengths
- [x] Strategy A smoke test passed on Colab T4 (`--num_frames 4 --epochs 1 --batch_size 1`) — loss 3.09→2.68 over 3 steps, checkpoint saved
- [x] Strategy A, B smoke tests passed. Strategy C hit a dtype crash (float32 activation vs
      bf16 weight in the frozen vision encoder) — root cause: `config/accelerate_config.yaml`
      only applies via `accelerate launch --config_file ...`, not plain `python train.py` (what
      Colab runs), so `Accelerator()` had no `mixed_precision` set and nothing enforced
      consistent dtype outside LoRA-wrapped layers. Fixed by setting `mixed_precision`
      explicitly in code (`train.py`), independent of how the script is launched.
- [x] All 4 strategies (A/B/C/D) smoke-tested clean on Colab T4. Trainable params: A=51.5M
      (0.62%), B=11.2M (0.13%), C=40.4M (0.48%), D=14.0M (0.17%) — all distinct, no config
      collisions. Phase 0 complete.

## Phase 1 — Data
- [x] Ego-Exo4D license granted, credentials received
- [x] Installed EgoExo downloader CLI (`pip install ego4d`), `aws configure` done
- [x] Downloaded `--parts metadata annotations` (10.3GiB, 5075 files) to `~/data/ego-exo4d`
      (outside repo) — verified complete: file count matches, size matches, no zero-byte
      files, all JSON parses cleanly, CLI itself reported 100%/"Everything downloaded"
- [x] Inspect real Ego-Exo4D annotation schema (keystep + atomic_descriptions, via docs) — see
      `src/data_prep/convert_egoexo4d.py` docstring for the confirmed field layout
- [x] Write Ego-Exo4D → `EgocentricHOIDataset` schema conversion script
      (`src/data_prep/convert_egoexo4d.py`) — mechanical parts (video lookup, sec→frame) are
      trustworthy; structured labels (tool/object/verb/state/safety-gear) are a naive heuristic
      split of `step_name`, flagged `needs_review` in `review_meta`, NOT ground truth yet
- [x] Ran the script against real `takes.json` + `keystep_train.json` — 0 takes missing an
      aria/rgb path across all 668 keystep-annotated takes; `find_egocentric_relative_path()`
      assumptions confirmed correct. (18/668 takes referenced in keystep have no matching
      entry in takes.json at all — likely dropped in a later dataset revision; script now
      tracks this via `takes_not_in_takes_json` stat instead of silently skipping)
- [x] Spot-checked heuristic labels against real keystep segments — confirmed noisy (~40-50%
      clean, rest broken on compound/clause step names; tool_detected/safety_gear_missing
      always empty, not present in keystep text at all)
- [x] Built LLM-assisted labeling (`src/data_prep/llm_label_segments.py`, Claude Haiku 4.5) —
      cross-references keystep segments against atomic_descriptions narrations in the same
      time window for richer extraction context. Wired into `convert_egoexo4d.py` via
      `--llm_labels_cache`, falls back to heuristic for uncovered segments.
- [x] Ran `llm_label_segments.py` on a 100-segment sample (5 takes, all "Covid-19 Rapid Antigen
      Test" — dict-order artifact, not real diversity). 0 failures. Real cost measured:
      118,634 in / 13,887 out tokens = $0.1881 for 100 segments (~$0.0019/segment). Quality
      confirmed better than heuristic — `tool_detected` populated 4x from real narration text,
      clean verb/object splits on compound step names.
- [x] Full-corpus labeling would cost ~$25 (13k+ segments, 668 takes) — decided against it.
      Defined curation criteria instead (below) so labeling only touches the curated subset
      (~600 segments, ~$1.15) — cheaper AND tests real scenario diversity on purpose.
- [x] Defined curation criteria ([ADR-0004](adr/0004-dataset-subset-curation.md)): 3 domains —
      Covid-19 Rapid Antigen Test (medical/fine-motor), Fix a Flat Tire (mechanical/tool-use),
      Cooking an Omelet (heat/state-change) — ~8-10 takes each, split by take_uid not segment.
      Surveyed all 17 scenarios in the corpus (668 takes total) to pick these deliberately.
- [x] Added `--scenarios`/`--max_takes_per_scenario` to both `convert_egoexo4d.py` and
      `llm_label_segments.py` so curated-subset selection is consistent across both scripts
- [x] First labeling run (649 segments, 30 takes, $1.21) had a real bug: `llm_label_segments.py`
      counted takes against the per-scenario cap by scenario name only, not usability — 2 of
      the 30 curated takes (both Covid-19 scenario) turned out to be among the 18/668 takes
      missing from `takes.json` entirely (found earlier), so they can never get a video path
      and the money spent labeling them was wasted (~$0.08). Fixed: `llm_label_segments.py`
      now takes an optional `--takes_json` and filters out unusable takes (missing from
      takes.json, or no aria/rgb path) *before* they consume a cap slot, reusing
      `convert_egoexo4d.py`'s `find_egocentric_relative_path()`.
- [ ] Rerun `llm_label_segments.py` with `--takes_json` added, to get a clean 30/30 usable-take
      cache (old `data/converted/llm_labels_cache.json` had 2 dead takes in it)
- [ ] Run `convert_egoexo4d.py` with the same `--scenarios`/`--max_takes_per_scenario` +
      `--llm_labels_cache` pointing at the labeled cache from the step above
- [ ] Only now download `--parts takes` for the ~24-30 curated takes' video (not the full corpus)
- [ ] Spot-check a sample of LLM-extracted labels — confirm quality improvement over heuristic
      before trusting as training data (ADR-0004)
- [ ] Build held-out eval split from the curated set, by take_uid (same split reused for all 4
      strategies)
- [ ] Sanity-check a handful of converted samples by hand

## Phase 2 — Real strategy runs (A, B, C, D) on rented GPU
Config/code already validated in Phase 0's smoke test — this phase is running the same 4
strategies for real, on the curated Ego-Exo4D subset once Phase 1 produces it.
- [ ] Pick cloud provider (see Open Questions)
- [ ] Run all 4 strategies on the curated subset — same split, seed, steps, batch size
- [ ] Log cost + wall time per strategy
- [ ] Save checkpoints + trainable-param counts (already known to differ correctly per strategy)

## Phase 4 — Evaluation
- [ ] Build eval harness (JSON validity, per-field accuracy, PONR F1, safety-gear P/R)
- [ ] Run all 4 checkpoints through harness on held-out split
- [ ] Sanity-check `serve.py` inference path against at least one trained checkpoint

## Phase 5 — Analysis & writeup
- [ ] Compare strategies: accuracy vs. trainable-param-% vs. $ cost
- [ ] Write up honest findings (including failures) — feed into [things-i-learned.md](things-i-learned.md)
- [ ] Finalize README (no "Built with Unsloth" fluff — lead with the research question)

## Open questions
- [ ] MiMo-VL vs Qwen2.5-VL — see ADR-0001
- [ ] Exact cloud provider (RunPod vs Vast.ai) — price/reliability check before committing
