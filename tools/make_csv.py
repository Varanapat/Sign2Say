"""รวม .npy + ไฟล์ split JSON -> CSV ที่ SPOTER อ่านได้ (54 จุดต่อเฟรม)"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
NPY_DIR = BASE / "pose_npy"
SPLIT_JSON = BASE / "tsl_one_s_dataset_i3d.json"
FPS, W, H = 29.97, 1280, 720             # ใช้เป็น metadata เท่านั้น

HEAD = {"nose": 0, "leftEye": 2, "rightEye": 5, "leftEar": 7, "rightEar": 8}
BODY = {"leftShoulder": 11, "rightShoulder": 12, "leftElbow": 13,
        "rightElbow": 14, "leftWrist": 15, "rightWrist": 16}
HAND = ["wrist", "thumbCMC", "thumbMP", "thumbIP", "thumbTip",
        "indexMCP", "indexPIP", "indexDIP", "indexTip",
        "middleMCP", "middlePIP", "middleDIP", "middleTip",
        "ringMCP", "ringPIP", "ringDIP", "ringTip",
        "littleMCP", "littlePIP", "littleDIP", "littleTip"]

# split แบบแบ่งตาม signer (signer-independent)
TEST_SIGNERS = {"03", "08", "22"}
VAL_SIGNERS = {"11", "25", "29"}


def put(row, name, xy):
    row[name + "_X"] = xy[:, 0].tolist()
    row[name + "_Y"] = xy[:, 1].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split-mode", choices=["json", "signer"], default="json")
    args = ap.parse_args()

    out_dir = BASE / "data" / f"split_{args.split_mode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    d = json.load(open(SPLIT_JSON, encoding="utf-8"))

    # label 0..183 เรียงตาม (ฟิลด์กลาง, เลขท้าย) ของแต่ละ gloss
    glosses = sorted(
        d, key=lambda g: tuple(g["instances"][0]["video_id"].split("_")[1:]))
    label_of = {g["gloss"]: k for k, g in enumerate(glosses)}
    with open(BASE / "data" / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_of, f, ensure_ascii=False, indent=1)

    rows = {"train": [], "val": [], "test": []}
    missing = []
    for g in d:
        for ins in g["instances"]:
            vid = ins["video_id"]
            path = NPY_DIR / f"{vid}.npy"
            if not path.exists():
                missing.append(vid)
                continue

            split = ins["split"]
            if args.split_mode == "signer":
                s = vid.split("_")[0]
                split = ("test" if s in TEST_SIGNERS
                         else "val" if s in VAL_SIGNERS else "train")

            a = np.load(path).reshape(-1, 75, 2)
            if len(a) == 0:
                missing.append(vid)
                continue
            pose, lh, rh = a[:, :33], a[:, 33:54], a[:, 54:75]

            row = {}
            for n, i in HEAD.items():
                put(row, n, pose[:, i])
            put(row, "neck", (pose[:, 11] + pose[:, 12]) / 2)   # กึ่งกลางไหล่
            for n, i in BODY.items():
                put(row, n, pose[:, i])
            for k, n in enumerate(HAND):
                put(row, f"{n}_left", lh[:, k])
                put(row, f"{n}_right", rh[:, k])

            row.update(video_id=vid, labels=label_of[g["gloss"]],
                       label_name=g["gloss"], split=split,
                       video_width=W, video_height=H, fps=FPS,
                       length=len(a) / FPS)
            rows[split].append(row)

    for s, r in rows.items():
        pd.DataFrame(r).to_csv(out_dir / f"tsl_{s}.csv", index=False)
        n_cls = len({x["labels"] for x in r})
        print(f"{s}: {len(r)} clips, {n_cls} classes")
    print("ไม่มีไฟล์ .npy:", len(missing), missing[:5])


if __name__ == "__main__":
    main()