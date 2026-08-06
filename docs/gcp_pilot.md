# GCP L4 — Strategy A pilot runbook

Goal: get a **real steps/sec** number for Strategy A on the curated Ego-Exo4D subset, so
time/cost for B/C/D can be extrapolated before committing to full runs (PRD §9). This is a
benchmark, not a full run — cap it with `--max_steps`.

## 0. Machine + image (the part that's easy to get wrong)

- **Machine type:** `g2-standard-4` — 1× NVIDIA L4 (24 GB), 4 vCPU, 16 GiB RAM. Cheapest L4.
  Bump to `g2-standard-8` only if host RAM becomes the limit during data loading.
- **Image:** an **accelerator-optimized Ubuntu** image (NVIDIA driver + CUDA preinstalled).
  **Do NOT use a Deep Learning VM image** — Google explicitly does not support DL VM images as
  boot disks on G2 instances; the create will fail or the driver won't match.
- **Region:** `us-central1` (Iowa), ~$0.71/hr on-demand for `g2-standard-4`.
- **Pricing:** **on-demand, not Spot.** train.py has no checkpoint/resume yet, so a Spot
  preemption loses the whole run. Switch to Spot only after resume support exists.
- **Disk:** 100 GB. Video ≈ 3.8 GB + 4-bit model cache + overhead.

Find the current accelerator-optimized Ubuntu image family (names change — don't hardcode a
stale one):

```bash
gcloud compute images list --project ubuntu-os-accelerator-images --filter="family~accelerator" \
  --format="table(name,family)"
```

Create the VM (substitute `IMAGE_FAMILY` from the list above):

```bash
gcloud compute instances create egoscribe-l4 \
  --zone=us-central1-a \
  --machine-type=g2-standard-4 \
  --accelerator=type=nvidia-l4,count=1 \
  --maintenance-policy=TERMINATE \
  --image-family=IMAGE_FAMILY \
  --image-project=ubuntu-os-accelerator-images \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-balanced
```

SSH in:

```bash
gcloud compute ssh egoscribe-l4 --zone=us-central1-a
```

Confirm the GPU is visible before spending time on setup:

```bash
nvidia-smi   # must show 1x L4, 24GB
```

## 1. Repo + environment

```bash
git clone <repo-url> egoscribe-vlm && cd egoscribe-vlm
python3 -m venv venv && source venv/bin/activate
pip install -r requirements-train.txt   # training stack ONLY, not the serving stack
```

Versions in `requirements-train.txt` are pinned to the Colab-validated set — do not bump.
If the image ships a preinstalled torch matched to its CUDA build and pip fights it, prefer the
image's torch (`pip install --no-deps torch==...` or drop torch from the install) rather than
forcing a mismatched CUDA wheel.

## 2. Data (the split files are gitignored — bring them + the video)

The 30 curated takes' video is not in the repo. Two options:

**A. Re-download on the VM (clean, reproducible).** Requires Ego-Exo4D license + the `egoexo`
CLI configured. Fetch metadata (gets `takes.json`), then resolve the exact 30 uids:

```bash
python -m src.data_prep.resolve_take_uids --takes_json /path/to/takes.json
# prints:  egoexo --parts downscaled_takes/448 --uids <30 uids>
```

Run that printed command. Then bring the two gitignored split files
(`data/converted/annotations_train.json`, `annotations_eval.json`) via `gcloud compute scp` from
local, and verify they match the committed manifest:

```bash
python -m src.data_prep.build_split_manifest --check   # must print "matches committed manifest"
```

**B. Transfer everything from local** (`gcloud compute scp` the ~3.8 GB video tree + the two
split JSONs). Skips egoexo setup on the VM.

Either way, `--video_dir` must point at the `takes/` root so `<take_name>/frame_aligned_videos/
downscaled/448/...` resolves.

## 3. Pilot benchmark — Strategy A, capped

Short capped run first to confirm the real-data pipeline end-to-end and get steps/sec:

```bash
python train.py \
  --strategy A \
  --annotations data/converted/annotations_train.json \
  --video_dir data/ego-exo4d/takes \
  --seed 3407 \
  --max_steps 30
```

Watch: the trainable-param count prints (should match Strategy A's expectation), loss decreases,
and the tqdm rate gives **it/s**. Real steps/sec × 858 steps/strategy → real time; × $0.71/hr →
real cost. Compare against the 5–10 hr/strategy rough guess (PRD §9) before launching full runs.

If VRAM is tight on the 24 GB L4, lower `--num_frames` or `--batch_size` (defaults 8 / 2).

## 4. Eval smoke (prove the harness runs on real checkpoints)

Full Strategy A run (drop `--max_steps`) saves an adapter to
`saved_egoscribe_adapters/strategy_A`. Smoke the eval harness on a few records:

```bash
python -m src.eval.evaluate \
  --annotations data/converted/annotations_eval.json \
  --video_dir data/ego-exo4d/takes \
  --adapter saved_egoscribe_adapters/strategy_A \
  --limit 3 \
  --out eval_out/strategy_A_smoke.json
```

Confirms adapter loading, generation, JSON parsing, and metric computation all work before
running the full 110-segment eval. If `load_adapter` errors, fall back to
`PeftModel.from_pretrained` in `src/eval/evaluate.py:load_model`.

## 5. Stop the meter

```bash
gcloud compute instances stop egoscribe-l4 --zone=us-central1-a   # stop between sessions
gcloud compute instances delete egoscribe-l4 --zone=us-central1-a # when fully done
```

A stopped VM still bills for the disk; delete when the pilot + runs are finished.
