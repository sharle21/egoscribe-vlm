# tasks.md — EgoScribe working checklist

Status legend: `[ ]` todo · `[~]` in progress · `[x]` done

## Phase 0 — Setup & code fixes
- [ ] `egodex_test.zip` (17GB, local) is the wrong dataset — confirm safe to delete once
      Ego-Exo4D access/download is sorted, to free disk space
- [~] Resolve model choice ([ADR-0001](adr/0001-model-selection.md)) — leaning Qwen2.5-VL, not fully closed
- [x] Fix label masking bug: pad `labels` with `-100`, not tokenizer pad id (`src/dataset.py`)
- [x] Switch `train.py` model loading from full BF16 to 4-bit QLoRA via Unsloth (`--strategy` flag implements ADR-0002 + ADR-0003)
- [x] Add Unsloth + bitsandbytes to `requirements.txt`
- [ ] `pip install -r requirements.txt` in a real GPU env and confirm Unsloth actually loads `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit`
- [ ] Prototype end-to-end pipeline on toy `data/samples/` on free tier (Colab/Kaggle) — confirm it runs before spending money, run all 4 `--strategy` values once each on the toy set as a smoke test

## Phase 1 — Data
- [ ] Confirm Ego-Exo4D license/access is actually granted
- [x] Inspect real Ego-Exo4D annotation schema (keystep + atomic_descriptions, via docs) — see
      `src/data_prep/convert_egoexo4d.py` docstring for the confirmed field layout
- [x] Write Ego-Exo4D → `EgocentricHOIDataset` schema conversion script
      (`src/data_prep/convert_egoexo4d.py`) — mechanical parts (video lookup, sec→frame) are
      trustworthy; structured labels (tool/object/verb/state/safety-gear) are a naive heuristic
      split of `step_name`, flagged `needs_review` in `review_meta`, NOT ground truth yet
- [ ] Run the script once against real downloaded `takes.json` + `keystep_train.json` and
      confirm `find_egocentric_relative_path()`'s assumed `frame_aligned_videos[cam][stream]`
      layout actually matches (field names were confirmed via docs/search, not a real file —
      treat as unverified until run once)
- [ ] Human review pass over a sample of `review_meta.needs_review=True` records — decide
      whether naive verb/object split is good enough or needs an LLM-assisted labeling pass
- [ ] Define curation criteria for the fixed subset (task diversity, interaction-type balance),
      then rerun conversion with `--scenario`/`--limit` to produce the actual curated set
- [ ] Build held-out eval split from the curated set (same split reused for all 4 strategies)
- [ ] Sanity-check a handful of converted samples by hand

## Phase 2 — Strategy A: Full LoRA (baseline)
- [ ] Configure `LoraConfig` for all target modules
- [ ] Run on rented GPU, log cost + wall time
- [ ] Save checkpoint + trainable-param count

## Phase 3 — Strategies B, C, D
- [ ] Strategy B: vision-encoder only
- [ ] Strategy C: language-decoder only
- [ ] Strategy D: attention + MLP hybrid
- [ ] Same subset, same steps/epochs, same seed policy as baseline

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
