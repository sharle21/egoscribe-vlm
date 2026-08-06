"""Map the curated split's take_names -> take_uids and print the egoexo download command.

The split manifest records take_names (the first path component of each video_file), but the
Ego-Exo4D `egoexo` CLI downloads by take_uid. This resolves one to the other via the corpus
takes.json (which egoexo fetches as its metadata part) so the GCP VM can pull exactly the 30
curated takes' downscaled/448 video — no full-corpus, no full-resolution.

  python -m src.data_prep.resolve_take_uids --takes_json /path/to/takes.json
"""
import argparse
import json

MANIFEST = "metadata/split_manifest.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--takes_json", required=True, help="Ego-Exo4D corpus takes.json (from egoexo metadata).")
    p.add_argument("--manifest", default=MANIFEST)
    args = p.parse_args()

    manifest = json.load(open(args.manifest))
    names = sorted(set(manifest["train"]["takes"]) | set(manifest["eval"]["takes"]))

    uid_by_name = {t["take_name"]: t["take_uid"] for t in json.load(open(args.takes_json))}
    uids, missing = [], []
    for name in names:
        uid = uid_by_name.get(name)
        (uids if uid else missing).append(uid or name)

    if missing:
        raise SystemExit(f"{len(missing)} take_names not found in takes.json: {missing}")

    print(f"# {len(uids)} curated take_uids")
    print(f"egoexo --parts downscaled_takes/448 --uids {' '.join(uids)}")


if __name__ == "__main__":
    main()
