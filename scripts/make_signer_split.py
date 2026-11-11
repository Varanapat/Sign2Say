"""Build signer-independent splits for TSL-ONE-S.

The released tsl_one_s_dataset_i3d.json puts every signer in train, so a model
can be tested on people it has already seen. This script regroups the same
clips so that no signer appears in more than one of train / val / test.

Video IDs follow [SignerID]_[CategoryID]_[GlossID] (TSL-ONE-S paper, Sec. III),
so the signer is the first field of video_id.

Signers are split into K groups of roughly equal clip count. Rotation f uses
group f as test, group f+1 as val, and the rest as train. With K=10 this gives
about 8:1:1, the ratio the paper reports; running all K rotations gives
leave-group-out cross-validation where every signer is tested once.

Output files keep the original JSON format, so existing I3D / WLASL-style
loaders read them unchanged.

Usage:
    python3 scripts/make_signer_split.py
    python3 scripts/make_signer_split.py --folds 5 --seed 0 --out splits
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def signer_of(video_id):
    return video_id.split("_")[0]


def make_groups(clips_per_signer, k, seed):
    """Assign signers to k groups, balancing clip counts greedily."""
    rng = random.Random(seed)
    signers = list(clips_per_signer)
    rng.shuffle(signers)  # random tie-breaking, stable for a given seed
    signers.sort(key=lambda s: clips_per_signer[s], reverse=True)

    groups = [[] for _ in range(k)]
    load = [0] * k
    for s in signers:
        g = min(range(k), key=lambda i: (load[i], len(groups[i])))
        groups[g].append(s)
        load[g] += clips_per_signer[s]
    return groups


def build_split(data, groups, fold):
    k = len(groups)
    split_of = {}
    for s in groups[fold]:
        split_of[s] = "test"
    for s in groups[(fold + 1) % k]:
        split_of[s] = "val"
    for i, g in enumerate(groups):
        if i not in (fold, (fold + 1) % k):
            for s in g:
                split_of[s] = "train"

    out = []
    for entry in data:
        instances = [
            {"video_id": inst["video_id"], "split": split_of[signer_of(inst["video_id"])]}
            for inst in entry["instances"]
        ]
        out.append({"gloss": entry["gloss"], "instances": instances})
    return out, split_of


def summarize(split_data, split_of):
    clips = Counter()
    glosses = defaultdict(set)
    for entry in split_data:
        for inst in entry["instances"]:
            clips[inst["split"]] += 1
            glosses[inst["split"]].add(entry["gloss"])
    signers = defaultdict(list)
    for s, sp in sorted(split_of.items()):
        signers[sp].append(s)

    # Hard guarantee: no signer in two splits.
    assert not (set(signers["train"]) & set(signers["val"]))
    assert not (set(signers["train"]) & set(signers["test"]))
    assert not (set(signers["val"]) & set(signers["test"]))

    unseen = sorted((glosses["val"] | glosses["test"]) - glosses["train"])
    return clips, signers, unseen


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--input", default=ROOT / "tsl_one_s_dataset_i3d.json", type=Path)
    ap.add_argument("--out", default=ROOT / "splits", type=Path)
    ap.add_argument("--folds", default=10, type=int, help="number of signer groups (10 -> ~8:1:1)")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    clips_per_signer = Counter(
        signer_of(inst["video_id"]) for entry in data for inst in entry["instances"]
    )
    if args.folds < 3 or args.folds > len(clips_per_signer):
        raise SystemExit(f"--folds must be between 3 and {len(clips_per_signer)}")

    groups = make_groups(clips_per_signer, args.folds, args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"{len(data)} glosses, {sum(clips_per_signer.values())} clips, "
          f"{len(clips_per_signer)} signers -> {args.folds} groups (seed {args.seed})")
    for i, g in enumerate(groups):
        print(f"  group {i}: signers {sorted(g)}  clips {sum(clips_per_signer[s] for s in g)}")

    manifest = {"input": str(args.input.name), "folds": args.folds, "seed": args.seed,
                "groups": [sorted(g) for g in groups], "rotations": []}

    for f in range(args.folds):
        split_data, split_of = build_split(data, groups, f)
        clips, signers, unseen = summarize(split_data, split_of)
        path = args.out / f"tsl_one_s_signer_indep_fold{f}.json"
        path.write_text(json.dumps(split_data, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"\nfold {f} -> {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
        for sp in ("train", "val", "test"):
            print(f"  {sp:5s} clips {clips[sp]:5d}  signers {signers[sp]}")
        if unseen:
            print(f"  ! {len(unseen)} gloss(es) in val/test never seen in train: {unseen}")

        manifest["rotations"].append({
            "fold": f, "file": path.name,
            "signers": {sp: signers[sp] for sp in ("train", "val", "test")},
            "clips": {sp: clips[sp] for sp in ("train", "val", "test")},
            "glosses_missing_from_train": unseen,
        })

    (args.out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.folds} split files + manifest.json to {args.out}")


if __name__ == "__main__":
    main()
