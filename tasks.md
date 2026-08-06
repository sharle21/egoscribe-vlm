# tasks.md — EgoScribe working checklist

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Phase 0 — Setup & code fixes
- [x] Deleted `egodex_test.zip` (16GB, wrong dataset) and an abandoned full-resolution
      `~/data/ego-exo4d/takes/` download (21GB, superseded by `downscaled_takes/448`) after
      disk hit 99% full mid-download — freed 37GB (190Mi → 38GB available)
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
- [x] Reran with `--takes_json` — clean 30/30 usable takes (10/10/10 balanced), 682 segments,
      0 failures, $1.27. Correctly skipped 10 unusable takes across all 3 scenarios (found
      before spending on them, not after). `data/converted/llm_labels_cache.json` is now the
      real curated-subset label cache.
- [x] Downloaded video for the 30 curated takes only, via `egoexo --parts downscaled_takes/448
      --uids <30 uids>` (not full corpus, not full resolution — 3.79GiB total, matches
      `EgocentricHOIDataset`'s resolution cap). Hit two real problems along the way, both
      fixed: (1) disk filled to 99% mid-download from an abandoned full-res attempt, freed by
      deleting that + `egodex_test.zip`; (2) `--parts takes` alone pulls every camera stream
      per take (84GB for 30 clips) — `downscaled_takes/448` is the right, much smaller part.
- [x] Ran `convert_egoexo4d.py` against real downloaded video — found and fixed two real path
      bugs: (1) `rel_path` from `takes.json` already includes the `frame_aligned_videos/`
      prefix, but the script joined it again, producing a path that could never match any real
      file (silent until video actually existed to test against); (2) the scenario filter
      checked `takes.json`'s `scenario` field, but that field doesn't exist there — it only
      exists on `keystep_train.json`'s records (`takes.json` has `task_name`/`parent_task_name`
      instead), so the filter matched 0/668 takes with zero error, only caught via the stats
      counter. Both fixed; `--use_downscaled_448` flag added.
- [x] `data/converted/annotations.json` now exists: 682 real training records, exactly matching
      all 682 LLM-labeled segments (0 fallback to heuristic, 0 bad videos). This is the actual
      curated training set — ready to point `EgocentricHOIDataset` at.
- [x] Spot-checked 3 records (one per domain) — bike, covid, cooking. All 3 fully confirmed:
      bike-shop fisheye view with hands near a wheel matched "checking for damage"/"tire";
      overhead unboxing shot matched "unboxing"/"package/carton"; kitchen scene matched
      "turning on"/"stove" — the "lighter" tool wasn't visible in the single still frame first
      checked, but user manually scrubbed the actual video (~0:32-0:36) and confirmed it's
      there. Labels track real video content, no discrepancies found.
- [x] Built held-out train/eval split (`src/data_prep/split_curated_dataset.py`), by take_uid,
      stratified per scenario (8/2 takes per domain, seed 42): 572 train segments / 24 takes,
      110 eval segments / 6 takes. Verified zero take overlap between splits. Written to
      `data/converted/annotations_train.json` / `annotations_eval.json` — this exact pair is
      reused identically across all 4 strategy runs (Phase 2). Phase 1 (data) complete.

## Phase 2 — Real strategy runs (A, B, C, D) on rented GPU
Config/code already validated in Phase 0's smoke test — this phase is running the same 4
strategies for real, on the curated Ego-Exo4D subset (Phase 1, done).
- [x] Picked provider: Google Cloud, G2 instance w/ 1x NVIDIA L4 (24GB), paid via $300/90-day
      free-trial credit (upgraded to paid billing to unlock GPU access, no charge yet)
- [~] Setting up GCP VM: `gcloud compute instances create egoscribe-l4` (G2, L4, PyTorch image,
      100GB disk) — walking through gcloud CLI install/auth/VM creation/SSH
- [x] Pinned `torch==2.11.0` / `transformers==5.5.0` / `unsloth==2026.8.4` in
      `requirements-train.txt` — exact versions confirmed clean across all 4 strategies on the
      Colab smoke test, so the GCP VM doesn't silently pull newer versions that could
      reintroduce a bug we already fixed around.
- [ ] On the VM: clone repo, `pip install -r requirements-train.txt` — CHECK first whether this
      conflicts with the Deep Learning VM image's pre-installed torch (matched to its own CUDA
      build); may need `--no-deps` on torch or to trust the image's preinstalled version instead
      of reinstalling. Pull curated dataset + video (either re-run `egoexo --uids <30 curated
      uids>` on the VM, or transfer from local).
- [ ] Benchmark Strategy A for ~15-20 min — get real steps/sec, extrapolate real time/cost for
      all 4 strategies before committing to full runs (training-time estimate so far is a rough
      guess extrapolated from Colab T4, could be off 2x either way)
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

## Open questions (resolved, kept for record)
- [x] MiMo-VL vs Qwen2.5-VL — Qwen2.5-VL, see ADR-0001 (Accepted)
- [x] Cloud provider — Google Cloud L4 + $300 free credit, over RunPod/Vast.ai (see Phase 2)
