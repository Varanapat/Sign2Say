"""Training loop ของ Part C (RGB) — กติกาเดียวกับ Part B: เลือก checkpoint จาก val macro-F1, early stopping,
ไฟล์ผลรูปแบบกลางใน results/<exp>_fold<k>_seed<s>/

ต้องตัดเฟรมก่อน: python tools/extract_frames.py

    python -m experiments.train_rgb --model C3 --fold 4 --seed 0
    python -m experiments.train_rgb --model C1 --exp smoke_C1 --epochs 1 --no_pretrained --limit 64   # ทดสอบ

budget ร่วม (README_brief §3.4 Part C): 30 epoch, ทุกโมเดล 16 เฟรม × 112², AdamW lr 1e-4 (fine-tune ทั้งโมเดล)
"""
import argparse
import csv
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from experiments import data as D
from experiments import metrics as M
from experiments import results as R
from experiments.models.rgb import RGB_MODELS
from experiments.rgb_data import CROP, RGBClips, rgb_split, worker_init
from experiments.train import get_device


@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = []
    for x, _ in loader:
        out.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(RGB_MODELS))
    ap.add_argument("--exp")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers (อ่านไฟล์ + aug)")
    ap.add_argument("--no_pretrained", action="store_true", help="ไม่โหลด pretrained weight (ใช้ทดสอบ)")
    ap.add_argument("--limit", type=int, default=0, help="ใช้แค่ N คลิปแรกของแต่ละ split (ใช้ทดสอบ)")
    a = ap.parse_args()
    exp = a.exp or a.model

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = get_device(a.device)
    builder, level, norm = RGB_MODELS[a.model]
    split, missing = rgb_split(a.fold)
    if a.limit:
        split = {k: (ids[:a.limit], y[:a.limit]) for k, (ids, y) in split.items()}
    (tr_ids, tr_y), (va_ids, va_y), (te_ids, te_y) = split["train"], split["val"], split["test"]
    if not tr_ids:
        raise SystemExit("ไม่พบไฟล์เฟรม — รัน python tools/extract_frames.py ก่อน")

    model = builder(D.NUM_CLASSES, not a.no_pretrained).to(device)
    n_params = M.count_params(model)
    out = R.run_dir(exp, a.fold, a.seed)
    print(f"{exp} ({a.model}, {level}, {norm}) | fold {a.fold} seed {a.seed} | device {device} | "
          f"params {n_params:,} | train {len(tr_y)} val {len(va_y)} test {len(te_y)} "
          f"(ไม่มีไฟล์เฟรม {len(missing)}) -> {out}", flush=True)

    kw = dict(num_workers=a.workers, persistent_workers=a.workers > 0, worker_init_fn=worker_init)
    g = torch.Generator().manual_seed(a.seed)
    tl = DataLoader(RGBClips(tr_ids, tr_y, norm, train=True, seed=a.seed), batch_size=a.batch_size,
                    shuffle=True, generator=g, drop_last=False, **kw)
    vl = DataLoader(RGBClips(va_ids, va_y, norm), batch_size=16, **kw)
    sl = DataLoader(RGBClips(te_ids, te_y, norm), batch_size=16, **kw)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    crit = torch.nn.CrossEntropyLoss()
    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_top1", "val_macro_f1", "seconds"])

    best, best_ep, t_all = (-1.0, -1.0), 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)
        vm = M.evaluate(predict_probs(model, vl, device), va_y, D.NUM_CLASSES)
        dt = time.time() - t0
        mark = ""
        if (vm["macro_f1"], vm["top1"]) > best:
            best, best_ep = (vm["macro_f1"], vm["top1"]), ep
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out / "best.pth")
            mark = "  *best"
        print(f"[{ep:3d}] loss {tot / n:.4f}  train {correct / n:.2%}  val top1 {vm['top1']:.2%}  "
              f"F1 {vm['macro_f1']:.4f}  ({dt:.0f}s){mark}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tot / n:.5f}", f"{correct / n:.5f}",
                                    f"{vm['top1']:.5f}", f"{vm['macro_f1']:.5f}", f"{dt:.1f}"])
        if ep - best_ep >= a.patience:
            print(f"early stopping: val macro-F1 ไม่ดีขึ้น {a.patience} epoch")
            break
    train_minutes = (time.time() - t_all) / 60

    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu", weights_only=True))
    model.to(device)
    pv, pt = predict_probs(model, vl, device), predict_probs(model, sl, device)
    val_m, test_m = M.evaluate(pv, va_y, D.NUM_CLASSES), M.evaluate(pt, te_y, D.NUM_CLASSES)
    R.write_predictions(out / "pred_val.csv", va_ids, va_y, pv)
    R.write_predictions(out / "pred_test.csv", te_ids, te_y, pt)

    model.eval()
    x1 = next(iter(sl))[0][:1].to(device)
    lat = M.latency_ms(model, x1, device)
    R.write_summary(
        out / "summary.json", exp=exp, level=level, modality="rgb", fold=a.fold, seed=a.seed,
        input=f"16 frames x {CROP}² RGB (letterbox, {norm} norm)", params=n_params, best_epoch=best_ep,
        val=val_m, test=test_m, train_minutes=train_minutes, latency_ms_b1=lat,
        device=R.device_name(device), model=a.model,
        hparams={k: getattr(a, k) for k in ("epochs", "patience", "batch_size", "lr", "weight_decay",
                                            "no_pretrained", "limit")})
    print(f"\nbest epoch {best_ep} | val top1 {val_m['top1']:.2%} F1 {val_m['macro_f1']:.4f} | "
          f"test top1 {test_m['top1']:.2%} top5 {test_m['top5']:.2%} F1 {test_m['macro_f1']:.4f} | "
          f"{train_minutes:.1f} min | latency {lat:.2f} ms\n-> {out}")


if __name__ == "__main__":
    main()
