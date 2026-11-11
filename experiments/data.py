"""โหลด landmark cache ของทีม (C54, T=64) แล้วแบ่ง train/val/test ตาม fold

cache: data/tsl1s_c54_T64.npz  (คัดลอกมาจาก my_work/dataset/cache/)
  X       [N, 64, 54, 7]  = xyz (0-2), velocity (3-5), valid mask (6)  — normalize แล้ว
  label   ชื่อ gloss        -> แปลงเป็น id ด้วย data/label_map.json (มาตรฐานกลาง)
  clip_id "tsl1s:<video_id>"

split: splits/tsl_one_s_signer_indep_fold<k>.json — คลิปที่ไม่อยู่ในไฟล์ split จะถูกตัดออก
"""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "tsl1s_c54_T64.npz"
LABEL_MAP = ROOT / "data" / "label_map.json"
SPLITS = ROOT / "splits"
NUM_CLASSES = 184

# ชื่อชุด channel -> index ใน cache
CHANNELS = {
    "all": [0, 1, 2, 3, 4, 5, 6],
    "xyz": [0, 1, 2],
    "xy": [0, 1],
    "xy_mask": [0, 1, 6],
    "xyz_vel": [0, 1, 2, 3, 4, 5],
}


@dataclass
class Split:
    X: np.ndarray        # [n, T, J, C] float32
    y: np.ndarray        # [n] int64
    ids: list            # video_id ต่อคลิป (ลำดับเดียวกับ X)


def fold_json(fold: int) -> Path:
    return SPLITS / f"tsl_one_s_signer_indep_fold{fold}.json"


def load_split_of(fold: int) -> dict:
    """{video_id: 'train' | 'val' | 'test'}"""
    data = json.loads(fold_json(fold).read_text(encoding="utf-8"))
    return {ins["video_id"]: ins["split"] for g in data for ins in g["instances"]}


def load_label_map() -> dict:
    return json.loads(LABEL_MAP.read_text(encoding="utf-8"))


def parse_channels(spec: str) -> list:
    """'all' / 'xy' / ... หรือ index คั่นด้วย comma เช่น '0,1,6'"""
    if spec in CHANNELS:
        return CHANNELS[spec]
    return [int(c) for c in spec.split(",")]


def load_landmarks(fold: int, channels: str = "all", cache: Path = CACHE) -> dict:
    """-> {'train': Split, 'val': Split, 'test': Split}"""
    z = np.load(cache)
    label_map = load_label_map()
    split_of = load_split_of(fold)

    ids = np.array([c.split(":", 1)[-1] for c in z["clip_id"]])
    y = np.array([label_map[g] for g in z["label"]], dtype=np.int64)
    X = z["X"][..., parse_channels(channels)].astype(np.float32)
    split = np.array([split_of.get(v, "") for v in ids])

    out = {}
    for s in ("train", "val", "test"):
        m = split == s
        out[s] = Split(X[m], y[m], ids[m].tolist())
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="ตรวจจำนวนคลิปต่อ split")
    ap.add_argument("--fold", type=int, default=4)
    a = ap.parse_args()
    d = load_landmarks(a.fold)
    for s, sp in d.items():
        signers = sorted({v.split("_")[0] for v in sp.ids})
        print(f"{s:5s} {len(sp.y):5d} clips  {len(set(sp.y.tolist()))} classes  X{tuple(sp.X.shape)}  "
              f"signers {signers}")
