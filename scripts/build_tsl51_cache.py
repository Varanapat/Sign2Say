"""Build a preprocessed TSL-51 ISLR cache (original clips only, no pre-made augment copies).

Output: dataset/cache/tsl51_c54_T64.npz with
  X        [N, 64, 54, 7] float32
  sign_id  [N] str      label (read from sign_id: sign_clean is empty for 448 null_act rows)
  signer   [N] str      expert_01 / expert_02 / web_<source> / user01
  group    [N] str      original_video_id (expert) or video_id (user)
  part     [N] str      expert / user
  clip_id  [N] str

Metadata landmark paths do not match the extracted folder names, so they are mapped
(verified 100% match, Data_Pipeline_Sign2Say.md section 2 stage 1).
"""

import csv
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from landmarks import preprocess, read_tsl51_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TSL51 = ROOT / "Source(Paper)" / "tsl51"
OUT = ROOT / "dataset" / "cache" / "tsl51_c54_T64.npz"
DIR_MAP = {"expert_primary_01": TSL51 / "landmarks" / "landmarks",
           "expert_primary_02": TSL51 / "landmarks" / "landmarks 2"}


def rows(name):
    return list(csv.DictReader(open(TSL51 / "metadata" / name, encoding="utf-8-sig")))


def main():
    scraped = zipfile.ZipFile(TSL51 / "landmarks" / "expert_scraped.zip")
    X, meta, quarantine = [], [], []

    def add(text, w, h, info):
        x = preprocess(read_tsl51_csv(text, w, h))
        if x is None:
            quarantine.append({**info, "reason": "no shoulders"})
            return
        X.append(x)
        meta.append(info)

    for r in rows("expert_metadata.csv"):
        if r["is_augmented"] != "False":
            continue
        if int(float(r["frames_extracted"] or 0)) < 15:
            quarantine.append({"clip_id": "tsl51e:" + r["video_id"], "reason": "fewer than 15 frames"})
            continue
        grp, base = r["landmark_path"].split("/")[1], Path(r["landmark_path"]).name
        if grp == "expert_scraped":
            text = scraped.read("landmarks/" + base).decode("utf-8")
            signer = "web_" + r["source"]
        else:
            text = (DIR_MAP[grp] / base).read_text(encoding="utf-8")
            signer = r["signer_name"]
        add(text, float(r["width_extracted"]), float(r["height_extracted"]),
            {"clip_id": "tsl51e:" + r["video_id"], "sign_id": r["sign_id"], "signer": signer,
             "group": r["original_video_id"], "part": "expert"})

    for r in rows("user_sign_metadata.csv"):
        text = (TSL51 / r["landmark_path"]).read_text(encoding="utf-8")
        add(text, float(r["width_extracted"]), float(r["height_extracted"]),
            {"clip_id": "tsl51u:" + r["video_id"], "sign_id": r["sign_id"], "signer": "user01",
             "group": r["video_id"], "part": "user"})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT, X=np.stack(X), **{k: np.array([m[k] for m in meta]) for k in meta[0]})
    (OUT.parent / "tsl51_quarantine.json").write_text(json.dumps(quarantine, ensure_ascii=False, indent=1))
    print(f"saved {len(X)} clips -> {OUT.relative_to(ROOT)}; quarantined {len(quarantine)}")
    print(Counter((m["part"], m["signer"]) for m in meta))
    ex = {m["sign_id"] for m in meta if m["part"] == "expert"}
    us = {m["sign_id"] for m in meta if m["part"] == "user"}
    print("classes expert", len(ex), "user", len(us), "user-only", us - ex)


if __name__ == "__main__":
    main()
