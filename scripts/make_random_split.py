"""Random-matched splits for TSL-ONE-S (Data_Pipeline_Sign2Say.md section 5.2, RQ5).

For each signer-independent fold k, build a random split with the *same number of
val/test clips per gloss* as fold k, but drawn from any signer. Only the signer
separation differs, so (random - SI) isolates the effect of seeing the test signers
during training.

Output: splits/random_matched/tsl_one_s_random_fold{k}.json (same JSON format).
"""

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "splits" / "random_matched"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for k in range(10):
        si = json.loads((ROOT / "splits" / f"tsl_one_s_signer_indep_fold{k}.json").read_text(encoding="utf-8"))
        rng = random.Random(1000 + k)
        out, n_same_signer = [], 0
        for entry in si:
            need = defaultdict(int)
            for inst in entry["instances"]:
                need[inst["split"]] += 1
            ids = [inst["video_id"] for inst in entry["instances"]]
            rng.shuffle(ids)
            split = (["test"] * need["test"] + ["val"] * need["val"] +
                     ["train"] * (len(ids) - need["test"] - need["val"]))
            out.append({"gloss": entry["gloss"],
                        "instances": [{"video_id": v, "split": s} for v, s in zip(ids, split)]})
        test_signers = {i["video_id"][:2] for e in out for i in e["instances"] if i["split"] == "test"}
        train_signers = {i["video_id"][:2] for e in out for i in e["instances"] if i["split"] == "train"}
        (OUT / f"tsl_one_s_random_fold{k}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                             encoding="utf-8")
        counts = defaultdict(int)
        for e in out:
            for i in e["instances"]:
                counts[i["split"]] += 1
        print(f"fold {k}: {dict(counts)}; test signers also in train: "
              f"{len(test_signers & train_signers)}/{len(test_signers)}")


if __name__ == "__main__":
    main()
