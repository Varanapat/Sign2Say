r"""
วัดผล checkpoint ของ SPOTER บน test set: top-1, top-5, macro-F1, คู่คำที่สับสนบ่อย
(วางในโฟลเดอร์ spoter ที่เดียวกับ train.py แล้วรันจากโฟลเดอร์นั้น)

ใช้:
  # ให้เลือก checkpoint_v_* ล่าสุดของ experiment (= ตัวที่ val ดีที่สุด) อัตโนมัติ
  python eval_ckpt.py --exp try5_accum16

  # หรือระบุไฟล์เอง
  python eval_ckpt.py --ckpt out-checkpoints\try5_accum16\checkpoint_v_3.pth

  # ดูว่ามี checkpoint อะไรบ้าง
  python eval_ckpt.py --exp try5_accum16 --list
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.spoter_model import SPOTER


def find_ckpt(exp, kind, list_only):
    files = sorted(Path("out-checkpoints", exp).glob(f"checkpoint_{kind}_*.pth"),
                   key=lambda p: p.stat().st_mtime)
    if list_only or not files:
        print(f"checkpoint ใน out-checkpoints\\{exp}:")
        for p in sorted(Path("out-checkpoints", exp).glob("*.pth"), key=lambda p: p.stat().st_mtime):
            print(f"  {p.name:25s} {p.stat().st_mtime_ns // 10**9}")
    if not files:
        raise SystemExit(f"ไม่พบ checkpoint_{kind}_*.pth")
    return files[-1]  # ไฟล์ล่าสุด = ค่าดีที่สุดที่ถูกบันทึก


def load_model(path, num_classes, hidden_dim):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        model = obj
    else:
        state = obj.get("model_state_dict", obj) if isinstance(obj, dict) else obj
        model = SPOTER(num_classes=num_classes, hidden_dim=hidden_dim)
        model.load_state_dict(state)
    return model.to("cpu").eval()


def load_names(path, num_classes):
    p = Path(path)
    if not p.exists():
        return {i: str(i) for i in range(num_classes)}
    m = json.loads(p.read_text(encoding="utf-8"))
    if all(str(k).lstrip("-").isdigit() for k in m.keys()):
        return {int(k): str(v) for k, v in m.items()}
    return {int(v): str(k) for k, v in m.items()}


def macro_f1(y, p, n):
    f1s = []
    for c in range(n):
        tp = int(np.sum((p == c) & (y == c)))
        fp = int(np.sum((p == c) & (y != c)))
        fn = int(np.sum((p != c) & (y == c)))
        if tp + fp + fn == 0:
            continue  # คลาสที่ไม่มีใน test และไม่ถูกทาย
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", help="ชื่อ experiment (โฟลเดอร์ใน out-checkpoints)")
    ap.add_argument("--ckpt", help="ระบุไฟล์ checkpoint เอง")
    ap.add_argument("--kind", default="v", choices=["v", "t"], help="v = เลือกจาก validation (ใช้รายงาน)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--csv", default=r"..\data\split_signer\tsl_test.csv")
    ap.add_argument("--label_map", default=r"..\data\label_map.json")
    ap.add_argument("--num_classes", type=int, default=184)
    ap.add_argument("--hidden_dim", type=int, default=108)
    ap.add_argument("--top_confused", type=int, default=15)
    a = ap.parse_args()

    if not a.ckpt and not a.exp:
        raise SystemExit("ใส่ --exp หรือ --ckpt")
    ckpt = Path(a.ckpt) if a.ckpt else find_ckpt(a.exp, a.kind, a.list)
    print(f"checkpoint: {ckpt}")
    print(f"test csv  : {a.csv}")

    model = load_model(ckpt, a.num_classes, a.hidden_dim)
    ds = CzechSLRDataset(a.csv)
    loader = DataLoader(ds, shuffle=False)

    ys, preds, top5 = [], [], 0
    with torch.no_grad():
        for x, y in loader:
            out = model(x.squeeze(0)).flatten()
            yi = int(y.flatten()[0])
            ys.append(yi)
            preds.append(int(out.argmax()))
            top5 += int(yi in out.topk(5).indices.tolist())
    y, p = np.array(ys), np.array(preds)
    n = len(y)

    acc = float((y == p).mean())
    f1 = macro_f1(y, p, a.num_classes)
    print(f"\nจำนวนคลิป test : {n}")
    print(f"Top-1 accuracy : {acc:.2%}  ({int((y == p).sum())}/{n})")
    print(f"Top-5 accuracy : {top5 / n:.2%}")
    print(f"Macro-F1       : {f1:.4f}")
    print(f"ทายแตกต่างกัน   : {len(set(p.tolist()))} คลาส (ถ้าน้อยมาก = โมเดลทายซ้ำๆ)")

    names = load_names(a.label_map, a.num_classes)
    pairs = Counter((int(t), int(q)) for t, q in zip(y, p) if t != q)
    print(f"\nคู่ที่ทายผิดบ่อยสุด {a.top_confused} อันดับ (จริง -> ทาย : จำนวน):")
    for (t, q), c in pairs.most_common(a.top_confused):
        print(f"  {names.get(t, t)} -> {names.get(q, q)} : {c}")

    out_csv = ckpt.with_name(ckpt.stem + "_test_predictions.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_id", "true_name", "pred_id", "pred_name", "correct"])
        for i, (t, q) in enumerate(zip(y, p)):
            w.writerow([i, t, names.get(int(t), t), q, names.get(int(q), q), int(t == q)])
    print(f"\nบันทึกผลทายรายคลิป: {out_csv}")


if __name__ == "__main__":
    main()
