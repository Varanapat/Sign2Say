"""เขียนไฟล์ผลตาม README_brief §3.6 เพื่อรวมผลกับทีมและทำ late fusion

results/<exp>_fold<k>_seed<s>/
  summary.json
  pred_val.csv, pred_test.csv   video_id,label,p0..p183 (หลัง softmax)
"""
import csv
import json
import platform
import subprocess
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"


def run_dir(exp: str, fold: int, seed: int, root: Path = RESULTS) -> Path:
    d = Path(root) / f"{exp}_fold{fold}_seed{seed}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_predictions(path: Path, ids, labels, probs: np.ndarray):
    n_cls = probs.shape[1]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["video_id", "label"] + [f"p{i}" for i in range(n_cls)])
        for vid, y, row in zip(ids, labels, probs):
            w.writerow([vid, int(y)] + [f"{v:.6g}" for v in row])


def write_summary(path: Path, *, exp, level, modality, fold, seed, input, params, best_epoch,
                  val, test, train_minutes, latency_ms_b1, device, **extra):
    s = {"exp": exp, "level": level, "modality": modality, "fold": fold, "seed": seed,
         "input": input, "params": int(params), "best_epoch": int(best_epoch),
         "val": _round(val), "test": _round(test),
         "train_minutes": round(float(train_minutes), 2),
         "latency_ms_b1": round(float(latency_ms_b1), 3), "device": device}
    s.update(extra)
    Path(path).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
    return s


def _round(m: dict) -> dict:
    return {k: round(float(v), 4) for k, v in m.items()}


def device_name(device) -> str:
    """ชื่ออุปกรณ์ที่อ่านง่าย สำหรับใส่ใน summary.json"""
    kind = getattr(device, "type", str(device))
    if kind == "cuda":
        import torch
        return torch.cuda.get_device_name(0)
    if platform.system() == "Darwin":
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        except OSError:
            chip = platform.machine()
        return f"{chip} ({kind.upper()})"
    return f"{platform.processor() or platform.machine()} ({kind})"
