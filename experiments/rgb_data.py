"""Dataset RGB ของ Part C: อ่าน data/rgb_T16_128/<video_id>.npy แล้วแบ่งตาม fold

train: สุ่ม crop 112² (ตำแหน่งเดียวกันทุกเฟรม) + ปรับความสว่าง/contrast เล็กน้อย — ไม่ flip (ผู้แสดงถนัดขวา)
val/test: center crop 112²
คืน tensor [T, 3, 112, 112] ที่ normalize ตามสถิติของ pretrained weights (imagenet หรือ kinetics)
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from experiments.data import fold_json, load_label_map, load_split_of

ROOT = Path(__file__).resolve().parent.parent
FRAMES = ROOT / "data" / "rgb_T16_128"
CROP = 112
NORM = {  # torchvision: ResNet (ImageNet) / video models (Kinetics-400)
    "imagenet": ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    "kinetics": ([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989]),
}


def rgb_split(fold: int, frames_dir: Path = FRAMES):
    """-> {'train': (ids, labels), ...} เฉพาะคลิปที่อยู่ใน fold JSON และมีไฟล์เฟรม"""
    label_of_gloss = load_label_map()
    split_of = load_split_of(fold)
    gloss_of = {ins["video_id"]: g["gloss"]
                for g in json.loads(fold_json(fold).read_text(encoding="utf-8")) for ins in g["instances"]}
    have = {p.stem for p in Path(frames_dir).glob("*.npy")}
    out = {}
    for s in ("train", "val", "test"):
        ids = sorted(v for v, sp in split_of.items() if sp == s and v in have)
        out[s] = (ids, np.array([label_of_gloss[gloss_of[v]] for v in ids], dtype=np.int64))
    missing = sorted(v for v in split_of if v not in have)
    return out, missing


class RGBClips(Dataset):
    def __init__(self, ids, labels, norm="imagenet", train=False, seed=0, frames_dir=FRAMES):
        self.ids, self.labels, self.train, self.dir = ids, labels, train, Path(frames_dir)
        mean, std = NORM[norm]
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        clip = np.load(self.dir / f"{self.ids[i]}.npy")              # [T, 128, 128, 3] uint8
        S = clip.shape[1]
        if self.train:
            y0, x0 = self.rng.integers(0, S - CROP + 1, size=2)
        else:
            y0 = x0 = (S - CROP) // 2
        clip = clip[:, y0:y0 + CROP, x0:x0 + CROP]
        x = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float() / 255.0   # [T,3,H,W]
        if self.train:
            b, c = self.rng.uniform(-0.1, 0.1), self.rng.uniform(0.9, 1.1)
            x = ((x - 0.5) * c + 0.5 + b).clamp(0, 1)
        return (x - self.mean) / self.std, int(self.labels[i])


def worker_init(worker_id):
    """DataLoader หลาย worker: ให้แต่ละ worker มี RNG ของตัวเอง (ไม่สุ่มซ้ำกัน)"""
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    ds.rng = np.random.default_rng([ds.seed, worker_id, torch.initial_seed() % 2**32])
