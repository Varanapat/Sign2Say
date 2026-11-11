"""Training loop กลางของ Part B — ทุกโมเดลใช้ data, budget, การเลือก checkpoint และไฟล์ผลเดียวกัน

    # รันจาก root ของ repo
    python -m experiments.train --model B0 --fold 4 --seed 0 --device mps
    python -m experiments.train --model B0 --epochs 1 --exp smoke_B0        # smoke test
    python -m experiments.train --model B1 --exp B1_aug --aug               # + augmentation แบบ SPOTER

กติกา (README_brief §3.4):
  - เลือก checkpoint จาก val macro-F1 (เสมอกันใช้ val top-1) — ไม่ใช้ test เลือก
  - early stopping: หยุดเมื่อ val macro-F1 ไม่ดีขึ้น --patience epoch
  - ผลอยู่ที่ results/<exp>_fold<k>_seed<s>/ (summary.json, pred_val.csv, pred_test.csv, log.csv, best.pth)
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
from experiments.augment import AugmentedLandmarks, select_features
from experiments.models import build


def get_device(name):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def predict_probs(model, X, device, batch_size=256):
    model.eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--exp", help="ชื่อการทดลอง (default = ชื่อโมเดล)")
    ap.add_argument("--fold", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", default="all", help="all / xyz / xy / xy_mask / xyz_vel หรือ index เช่น 0,1,6")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--device", default="auto", help="auto / cpu / mps / cuda")
    ap.add_argument("--aug", action="store_true",
                    help="augmentation แบบ SPOTER (rotate/perspective/squeeze/arm-joint, p=0.5) — เฉพาะ train")
    ap.add_argument("--aug_prob", type=float, default=0.5)
    ap.add_argument("--hand_local", action="store_true",
                    help="เพิ่ม 2 channel: พิกัดมือในกรอบของมือเอง (คล้าย hand normalization ของ SPOTER)")
    ap.add_argument("--noise_std", type=float, default=0.006,
                    help="gaussian noise หน่วยความกว้างไหล่ (ใช้เมื่อ --aug; 0 = ปิด)")
    a = ap.parse_args()
    exp = a.exp or a.model

    set_seed(a.seed)
    device = get_device(a.device)
    d = D.load_landmarks(a.fold, "all")              # aug ต้องใช้ 7 channel เต็ม (xyz + mask)
    ch = D.parse_channels(a.channels)
    tr, va, te = d["train"], d["val"], d["test"]
    va.X = select_features(va.X, ch, a.hand_local)
    te.X = select_features(te.X, ch, a.hand_local)
    input_shape = tuple(tr.X.shape[1:3]) + (len(ch) + 2 * a.hand_local,)

    model, level, modality = build(a.model, input_shape, D.NUM_CLASSES)
    model.to(device)
    n_params = M.count_params(model)
    out = R.run_dir(exp, a.fold, a.seed)
    print(f"{exp} ({a.model}, {level}) | fold {a.fold} seed {a.seed} | device {device} | "
          f"params {n_params:,} | train {len(tr.y)} val {len(va.y)} test {len(te.y)} | "
          f"input {input_shape} ({a.channels}{' + hand_local' if a.hand_local else ''}) | aug {a.aug} -> {out}")

    g = torch.Generator().manual_seed(a.seed)
    train_ds = AugmentedLandmarks(tr.X, tr.y, ch, aug=a.aug, seed=a.seed,
                                  prob=a.aug_prob, noise_std=a.noise_std if a.aug else 0.0,
                                  use_hand_local=a.hand_local)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    crit = torch.nn.CrossEntropyLoss()

    log_path = out / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_top1", "val_macro_f1", "seconds"])

    best = (-1.0, -1.0)
    best_ep, t_all = 0, time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(yb)
            correct += int((logits.argmax(1) == yb).sum().item())
            n += len(yb)

        vm = M.evaluate(predict_probs(model, va.X, device), va.y, D.NUM_CLASSES)
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

    # ---- ประเมินด้วย checkpoint ที่ val ดีที่สุด ----
    model.load_state_dict(torch.load(out / "best.pth", map_location="cpu", weights_only=True))
    model.to(device)
    pv, pt = predict_probs(model, va.X, device), predict_probs(model, te.X, device)
    val_m, test_m = M.evaluate(pv, va.y, D.NUM_CLASSES), M.evaluate(pt, te.y, D.NUM_CLASSES)
    R.write_predictions(out / "pred_val.csv", va.ids, va.y, pv)
    R.write_predictions(out / "pred_test.csv", te.ids, te.y, pt)

    model.eval()
    lat = M.latency_ms(model, torch.from_numpy(te.X[:1]).to(device), device)
    T, J, C = input_shape
    s = R.write_summary(
        out / "summary.json", exp=exp, level=level, modality=modality, fold=a.fold, seed=a.seed,
        input=f"{J} joints x {C} ch ({a.channels}{' + hand_local' if a.hand_local else ''}), T={T}", params=n_params, best_epoch=best_ep,
        val=val_m, test=test_m, train_minutes=train_minutes, latency_ms_b1=lat,
        device=R.device_name(device), model=a.model,
        hparams={k: getattr(a, k) for k in ("epochs", "patience", "batch_size", "lr", "weight_decay",
                                            "aug", "aug_prob", "noise_std", "hand_local", "channels")})
    print(f"\nbest epoch {best_ep} | val top1 {val_m['top1']:.2%} F1 {val_m['macro_f1']:.4f} | "
          f"test top1 {test_m['top1']:.2%} top5 {test_m['top5']:.2%} F1 {test_m['macro_f1']:.4f} | "
          f"{train_minutes:.1f} min | latency {lat:.2f} ms\n-> {out}")
    return s


if __name__ == "__main__":
    main()
