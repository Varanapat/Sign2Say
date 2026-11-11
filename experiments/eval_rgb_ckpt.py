"""แผนสำรอง: ประเมิน checkpoint ที่ดีที่สุด (best.pth) ของ run RGB ที่ยังเทรนไม่จบ / ถูกหยุดกลางคัน

best.pth ถูกบันทึกทุกครั้งที่ val macro-F1 ดีขึ้น จึงใช้ได้ทันทีแม้ run ยังไม่จบ
เขียน pred_val.csv, pred_test.csv, summary.json แบบเดียวกับ train_rgb.py (+ ระบุว่าเป็นผลจาก run ที่ไม่จบ)

    python -m experiments.eval_rgb_ckpt --model C3 --fold 4 --seed 0               # เขียนลง results/C3_fold4_seed0
    python -m experiments.eval_rgb_ckpt --model C3 --device cpu --out_exp C3_partial   # เขียนลงโฟลเดอร์ใหม่

ถ้า run เดิมยังรันอยู่ ให้หยุดก่อน (Ctrl+C) ไม่งั้นจะแย่ง GPU และไฟล์ผลอาจถูกเขียนทับตอน run เดิมจบ
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from experiments import data as D
from experiments import metrics as M
from experiments import results as R
from experiments.models.rgb import RGB_MODELS
from experiments.rgb_data import CROP, RGBClips, rgb_split
from experiments.train import get_device
from experiments.train_rgb import predict_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RGB_MODELS))
    ap.add_argument("--exp", help="ชื่อ run ที่มี best.pth (default = ชื่อโมเดล)")
    ap.add_argument("--out_exp", help="เขียนผลลงชื่อนี้แทน (default = เขียนทับ run เดิม)")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="ใช้ N คลิปแรก (ทดสอบเท่านั้น)")
    a = ap.parse_args()
    exp = a.exp or a.model
    src = R.RESULTS / f"{exp}_fold{a.fold}_seed{a.seed}"
    if not (src / "best.pth").exists():
        raise SystemExit(f"ไม่พบ {src / 'best.pth'}")

    # อ่าน log ของ run เดิม: epoch ที่เทรนไปแล้ว, best epoch (val F1 สูงสุด), เวลารวม
    with open(src / "log.csv") as f:
        rows = list(csv.DictReader(f))
    best_row = max(rows, key=lambda r: (float(r["val_macro_f1"]), float(r["val_top1"])))
    epochs_done, minutes = len(rows), sum(float(r["seconds"]) for r in rows) / 60

    device = get_device(a.device)
    builder, level, norm = RGB_MODELS[a.model]
    split, _ = rgb_split(a.fold)
    if a.limit:
        split = {k: (ids[:a.limit], y[:a.limit]) for k, (ids, y) in split.items()}
    (va_ids, va_y), (te_ids, te_y) = split["val"], split["test"]

    model = builder(D.NUM_CLASSES, False)
    model.load_state_dict(torch.load(src / "best.pth", map_location="cpu", weights_only=True))
    model.to(device).eval()
    vl = DataLoader(RGBClips(va_ids, va_y, norm), batch_size=16, num_workers=a.workers)
    sl = DataLoader(RGBClips(te_ids, te_y, norm), batch_size=16, num_workers=a.workers)
    pv, pt = predict_probs(model, vl, device), predict_probs(model, sl, device)
    val_m, test_m = M.evaluate(pv, va_y, D.NUM_CLASSES), M.evaluate(pt, te_y, D.NUM_CLASSES)

    out = R.run_dir(a.out_exp or exp, a.fold, a.seed)
    R.write_predictions(out / "pred_val.csv", va_ids, va_y, pv)
    R.write_predictions(out / "pred_test.csv", te_ids, te_y, pt)
    lat = M.latency_ms(model, next(iter(sl))[0][:1].to(device), device)
    note = (f"run ไม่จบ: เทรนไป {epochs_done} epoch แล้วหยุด (เวลาไม่พอ); ผลนี้มาจาก best.pth "
            f"(val F1 สูงสุดที่ epoch {best_row['epoch']}) ประเมินด้วย experiments/eval_rgb_ckpt.py")
    R.write_summary(
        out / "summary.json", exp=a.out_exp or exp, level=level, modality="rgb", fold=a.fold, seed=a.seed,
        input=f"16 frames x {CROP}² RGB (letterbox, {norm} norm)", params=M.count_params(model),
        best_epoch=int(best_row["epoch"]), val=val_m, test=test_m, train_minutes=minutes,
        latency_ms_b1=lat, device=R.device_name(device), model=a.model, partial=True,
        epochs_done=epochs_done, note=note)
    print(f"{note}\nval top1 {val_m['top1']:.2%} F1 {val_m['macro_f1']:.4f} | test top1 {test_m['top1']:.2%} "
          f"top5 {test_m['top5']:.2%} F1 {test_m['macro_f1']:.4f}\n-> {out}")


if __name__ == "__main__":
    main()
