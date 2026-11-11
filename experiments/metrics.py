"""metric ตาม README_brief §3.5: top-1, top-5, macro-F1, จำนวนพารามิเตอร์, latency"""
import statistics
import time

import numpy as np


def topk_acc(probs: np.ndarray, y: np.ndarray, k: int) -> float:
    top = np.argsort(-probs, axis=1)[:, :k]
    return float((top == y[:, None]).any(axis=1).mean())


def macro_f1(y: np.ndarray, p: np.ndarray, num_classes: int) -> float:
    """เฉลี่ย F1 ทุกคลาส ข้ามคลาสที่ไม่มีทั้งใน y และ p (เหมือน spoter/eval_ckpt.py)"""
    f1s = []
    for c in range(num_classes):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else 0.0


def evaluate(probs: np.ndarray, y: np.ndarray, num_classes: int) -> dict:
    p = probs.argmax(axis=1)
    return {"top1": float((p == y).mean()),
            "top5": topk_acc(probs, y, 5),
            "macro_f1": macro_f1(y, p, num_classes)}


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def latency_ms(forward, x1, device, warmup: int = 10, runs: int = 100) -> float:
    """median เวลา forward ของ batch 1 (ms) หลัง warm-up — forward(x) คือฟังก์ชันที่รับ tensor บน device"""
    import torch
    with torch.no_grad():
        for _ in range(warmup):
            forward(x1)
        _sync(device)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            forward(x1)
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(times))
