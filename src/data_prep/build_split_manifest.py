"""Emit a committable manifest of the train/eval split (reproducibility, ADR-0004).

The split files themselves (data/converted/annotations_{train,eval}.json) are gitignored — they
embed video paths and are regenerated, not source. But the *identity* of the split (which
take_uids landed in eval, how many segments, and a checksum of the exact file contents) must
survive so a fresh checkout on the GCP VM can prove it is training on the same split that
produced the reported numbers. This script distills the two split files into
metadata/split_manifest.json, which IS committed.

Verify a local split matches the committed manifest:
  python -m src.data_prep.build_split_manifest --check

Regenerate after an intentional re-split:
  python -m src.data_prep.build_split_manifest
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

TRAIN_FILE = "data/converted/annotations_train.json"
EVAL_FILE = "data/converted/annotations_eval.json"
MANIFEST_FILE = "metadata/split_manifest.json"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def summarize(path):
    records = json.load(open(path))
    # take_uid == first path component of video_file (see split_curated_dataset.py:53)
    take_counts = Counter(r["video_file"].split("/", 1)[0] for r in records)
    return {
        "file": path,
        "sha256": sha256_file(path),
        "num_segments": len(records),
        "num_takes": len(take_counts),
        "takes": dict(sorted(take_counts.items())),  # take_uid -> segment count
    }


def build():
    manifest = {"train": summarize(TRAIN_FILE), "eval": summarize(EVAL_FILE)}
    # Leakage guard: no take_uid may appear in both splits (the whole point of ADR-0004).
    overlap = set(manifest["train"]["takes"]) & set(manifest["eval"]["takes"])
    manifest["leakage_check"] = {"overlapping_takes": sorted(overlap), "clean": not overlap}
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if local split != committed manifest")
    args = parser.parse_args()

    manifest = build()
    if not manifest["leakage_check"]["clean"]:
        raise SystemExit(f"LEAKAGE: takes in both splits: {manifest['leakage_check']['overlapping_takes']}")

    if args.check:
        committed = json.load(open(MANIFEST_FILE))
        for split in ("train", "eval"):
            if committed[split]["sha256"] != manifest[split]["sha256"]:
                raise SystemExit(
                    f"MISMATCH: local {split} split sha256 {manifest[split]['sha256'][:12]} "
                    f"!= committed {committed[split]['sha256'][:12]}. Local split is NOT the "
                    f"one the manifest was built from."
                )
        print("Local split matches committed manifest.")
        return

    Path(MANIFEST_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    t, e = manifest["train"], manifest["eval"]
    print(f"Wrote {MANIFEST_FILE}")
    print(f"  train: {t['num_segments']} segments / {t['num_takes']} takes  sha {t['sha256'][:12]}")
    print(f"  eval:  {e['num_segments']} segments / {e['num_takes']} takes  sha {e['sha256'][:12]}")
    print(f"  leakage clean: {manifest['leakage_check']['clean']}")


if __name__ == "__main__":
    main()
