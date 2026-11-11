r"""
เทรน SPOTER แบบ batch จริง (ทุกคลิป resample เป็น 64 เฟรม) — เพื่อให้ GPU (DirectML) คุ้ม
ใช้ loader / normalization / augmentation และโมเดลของ SPOTER เดิมทั้งหมด ไม่ต้องแก้ไฟล์ของ SPOTER เพิ่ม

วางไฟล์นี้ในโฟลเดอร์ spoter (ที่เดียวกับ train.py, dml_adam.py, eval_ckpt.py) แล้วรันจากโฟลเดอร์นั้น

1) ตรวจว่า batched forward ให้ผลเท่ากับโมเดลเดิมทีละคลิป (ต้องผ่านก่อน)
     python train_batched.py --check

2) วัดความเร็ว 1 epoch เทียบ GPU กับ CPU
     python train_batched.py --exp speed_dml --device dml --epochs 1
     python train_batched.py --exp speed_cpu --device cpu --epochs 1

3) เทรนจริง
     python train_batched.py --exp b16_signer_aug_80 --device dml --epochs 80 --aug

บน macOS ใช้ --device mps (Apple Silicon) หรือ cpu
ผลลัพธ์อยู่ที่ out-batched/<exp>/  (best_val.pth, log.csv, test_predictions.csv, summary.txt)

4) เทรนบน fold ของทีม (README_brief) — ต้องสร้าง CSV ก่อน:
     python ../tools/make_csv.py --split-json ../splits/tsl_one_s_signer_indep_fold4.json
     python train_batched.py --exp A1_spoter --fold 4 --seed 0 --device mps --aug \
         --epochs 100 --patience 20 --select f1
   ใช้ CSV จาก ../data/split_fold4/ และเขียนผลรูปแบบกลางที่ ../results/A1_spoter_fold4_seed0/
   (summary.json, pred_val.csv, pred_test.csv + ไฟล์เดิม)

5) A2 = SPOTER + positional encoding ตามเวลา: เพิ่ม --pe learned (หรือ --pe sin)
"""
import argparse
import csv
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.spoter_model import SPOTER
from eval_ckpt import load_names, macro_f1

SPLIT_DIR = "../data/split_signer"   # ใช้ / ได้ทั้ง Windows และ macOS
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))            # ให้ import experiments (ของกลาง) ได้ — ต่อท้ายเพื่อไม่บัง package spoter


# ---------------------------------------------------------------- data
def resample_frames(x, target):
    """(T, 54, 2) -> (target, 54, 2) เลือกเฟรมเว้นระยะเท่ากัน (คลิปสั้น = เฟรมซ้ำ, คลิปยาว = ข้ามเฟรม)"""
    n = x.shape[0]
    idx = torch.linspace(0, n - 1, target).round().long()
    return x[idx]


def make_collate(target, noise_std):
    def collate(batch):
        xs, ys = [], []
        for x, y in batch:
            x = resample_frames(torch.as_tensor(x).float(), target)
            if noise_std > 0:
                x = x + torch.randn_like(x) * noise_std  # เหมือน GaussianNoise ของ SPOTER
            xs.append(x)
            ys.append(int(torch.as_tensor(y).flatten()[0]))
        return torch.stack(xs), torch.tensor(ys, dtype=torch.long)
    return collate


# ---------------------------------------------------------------- model
def batched_forward(model, x):
    """
    x: (B, T, 54, 2) -> logits (B, num_classes)
    ทำแบบเดียวกับ SPOTER.forward แต่รับหลายคลิป:
      เดิม: h = (T, 1, 108), class_query = (1, 1, d)
      ใหม่: h = (T, B, 108), class_query = (1, B, d)
    """
    B = x.shape[0]
    h = x.flatten(start_dim=2).permute(1, 0, 2).float()          # (T, B, 108)
    tgt = model.class_query.unsqueeze(1).expand(-1, B, -1)       # (1, B, d)
    out = model.transformer(model.position(h.shape[0]) + h, tgt)  # (1, B, d)
    return model.linear_class(out[0])                            # (B, C)


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":                       # Mac Apple Silicon (M1–M4)
        if not torch.backends.mps.is_available():
            raise SystemExit("MPS ใช้ไม่ได้ในเครื่องนี้ (Mac Intel?) — ใช้ --device cpu")
        print("ใช้ GPU: Apple MPS")
        return torch.device("mps")
    import torch_directml
    print("ใช้ GPU:", torch_directml.device_name(0))
    return torch_directml.device()


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ys, ps, top5 = [], [], 0
    for x, y in loader:
        out = batched_forward(model, x.to(device)).to("cpu")
        ys += y.tolist()
        ps += out.argmax(1).tolist()
        top5 += sum(int(t in row) for t, row in zip(y.tolist(), out.topk(5, dim=1).indices.tolist()))
    return np.array(ys), np.array(ps), top5


@torch.no_grad()
def predict_probs(model, loader, device):
    """-> (labels, softmax probs [n, C]) ตามลำดับใน loader (ต้องไม่ shuffle)"""
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        ps.append(torch.softmax(batched_forward(model, x.to(device)), dim=1).to("cpu"))
        ys += y.tolist()
    return np.array(ys), torch.cat(ps).numpy()


# ---------------------------------------------------------------- check
def check(a):
    torch.manual_seed(0)
    model = SPOTER(num_classes=a.num_classes, hidden_dim=a.hidden_dim, pe=a.pe, max_len=a.frames).eval()
    ds = CzechSLRDataset(a.train_csv, augmentations=False)
    xs = [resample_frames(torch.as_tensor(ds[i][0]).float(), a.frames) for i in range(6)]
    with torch.no_grad():
        single = torch.cat([model(x).reshape(1, -1) for x in xs])   # โมเดลเดิม ทีละคลิป
        batch = batched_forward(model, torch.stack(xs))               # แบบ batch
    diff = (single - batch).abs().max().item()
    print(f"ผลต่างสูงสุดระหว่างทีละคลิปกับ batch: {diff:.2e} ->",
          "ผ่าน" if diff < 1e-4 else "ไม่ผ่าน (ส่งผลนี้มาให้ดู)")


# ---------------------------------------------------------------- train
def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    g = torch.Generator(); g.manual_seed(a.seed)
    device = get_device(a.device)

    train_ds = CzechSLRDataset(a.train_csv, augmentations=a.aug)
    val_ds = CzechSLRDataset(a.val_csv, augmentations=False)
    test_ds = CzechSLRDataset(a.test_csv, augmentations=False)
    tl = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True, generator=g,
                    collate_fn=make_collate(a.frames, a.noise_std))
    vl = DataLoader(val_ds, batch_size=64, collate_fn=make_collate(a.frames, 0))
    sl = DataLoader(test_ds, batch_size=64, collate_fn=make_collate(a.frames, 0))
    print(f"train {len(train_ds)} | val {len(val_ds)} | test {len(test_ds)} | "
          f"batch {a.batch_size} | frames {a.frames} | aug {a.aug} | device {a.device} | "
          f"split {Path(a.train_csv).parent.name} | select {a.select} | pe {a.pe}")

    model = SPOTER(num_classes=a.num_classes, hidden_dim=a.hidden_dim, pe=a.pe, max_len=a.frames).to(device)
    if a.device == "dml":
        from dml_adam import DmlAdam
        opt = DmlAdam(model.parameters(), lr=a.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = torch.nn.CrossEntropyLoss()

    if a.fold is not None:
        from experiments import results as R
        out_dir = R.run_dir(a.exp, a.fold, a.seed)      # ../results/<exp>_fold<k>_seed<s>/
    else:
        out_dir = Path("out-batched", a.exp)
        out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_acc", "val_macro_f1", "seconds"])

    best_val, best_ep, t_all = -1.0, 0, time.time()
    best_key = (-1.0, -1.0)
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, correct, n = time.time(), 0.0, 0, 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = batched_forward(model, x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
            correct += int((out.argmax(1) == y).sum().item())
            n += len(y)
        yv, pv, _ = predict(model, vl, device)
        val_acc = float((yv == pv).mean())
        val_f1 = macro_f1(yv, pv, a.num_classes)
        dt = time.time() - t0
        mark = ""
        key = (val_f1, val_acc) if a.select == "f1" else (val_acc,)
        if key > best_key[:len(key)]:
            best_key = key + (0.0,) * (2 - len(key))
            best_val, best_ep = val_acc, ep
            # ย้ายไป CPU ก่อนบันทึก: tensor บน DirectML โหลดกลับด้วย weights_only=True ไม่ได้
            torch.save({k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                       out_dir / "best_val.pth")
            mark = "  *best"
        print(f"[{ep:3d}] loss {tot / n:.4f}  train {correct / n:.2%}  val {val_acc:.2%}  "
              f"F1 {val_f1:.4f}  ({dt:.0f}s){mark}", flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tot / n:.5f}", f"{correct / n:.5f}", f"{val_acc:.5f}",
                                    f"{val_f1:.5f}", f"{dt:.1f}"])
        if a.patience and ep - best_ep >= a.patience:
            print(f"early stopping: val ไม่ดีขึ้น {a.patience} epoch")
            break
    train_minutes = (time.time() - t_all) / 60

    # ---- test ด้วย checkpoint ที่ val ดีที่สุด ----
    state = torch.load(out_dir / "best_val.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    y, p, top5 = predict(model, sl, device)
    acc, f1 = float((y == p).mean()), macro_f1(y, p, a.num_classes)
    names = load_names(a.label_map, a.num_classes)

    lines = [
        f"experiment      : {a.exp}",
        f"best val        : {best_val:.2%} (epoch {best_ep})",
        f"Test top-1      : {acc:.2%} ({int((y == p).sum())}/{len(y)})",
        f"Test top-5      : {top5 / len(y):.2%}",
        f"Test macro-F1   : {f1:.4f}",
        f"เวลารวม         : {(time.time() - t_all) / 60:.1f} นาที",
        "", "คู่ที่ทายผิดบ่อยสุด (จริง -> ทาย : จำนวน):",
    ]
    for (t, q), c in Counter((int(t), int(q)) for t, q in zip(y, p) if t != q).most_common(15):
        lines.append(f"  {names.get(t, t)} -> {names.get(q, q)} : {c}")
    text = "\n".join(lines)
    print("\n" + text)
    (out_dir / "summary.txt").write_text(text, encoding="utf-8")

    with open(out_dir / "test_predictions.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx", "true_id", "true_name", "pred_id", "pred_name", "correct"])
        for i, (t, q) in enumerate(zip(y, p)):
            w.writerow([i, t, names.get(int(t), t), q, names.get(int(q), q), int(t == q)])

    if a.fold is not None:
        write_standard_results(a, model, device, vl, sl, out_dir, best_ep, train_minutes)
    print(f"\nบันทึกผลไว้ที่ {out_dir}")


def write_standard_results(a, model, device, vl, sl, out_dir, best_ep, train_minutes):
    """ไฟล์ผลรูปแบบกลางของทีม (README_brief §3.6) — ใช้ model ที่โหลด best checkpoint แล้ว"""
    import pandas as pd
    from experiments import metrics as M
    from experiments import results as R

    yv, pv = predict_probs(model, vl, device)
    yt, pt = predict_probs(model, sl, device)
    ids_v = pd.read_csv(a.val_csv, usecols=["video_id"])["video_id"].tolist()
    ids_t = pd.read_csv(a.test_csv, usecols=["video_id"])["video_id"].tolist()
    R.write_predictions(out_dir / "pred_val.csv", ids_v, yv, pv)
    R.write_predictions(out_dir / "pred_test.csv", ids_t, yt, pt)

    x1 = next(iter(sl))[0][:1].to(device)
    lat = M.latency_ms(lambda x: batched_forward(model, x), x1, device)
    R.write_summary(
        out_dir / "summary.json", exp=a.exp, level="L0" if a.pe == "none" else "L1", modality="landmark", fold=a.fold, seed=a.seed,
        input=f"54 joints x (x,y), T={a.frames} (SPOTER normalization), pe={a.pe}", params=M.count_params(model),
        best_epoch=best_ep, val=M.evaluate(pv, yv, a.num_classes), test=M.evaluate(pt, yt, a.num_classes),
        train_minutes=train_minutes, latency_ms_b1=lat, device=R.device_name(device), model="SPOTER",
        hparams={k: getattr(a, k) for k in ("epochs", "patience", "batch_size", "lr", "aug", "pe",
                                            "noise_std", "frames", "select")})


def resolve_csv_paths(a):
    """--train/val/test_csv ที่ไม่ได้ระบุ -> เอาจาก --split_dir, ../data/split_fold<k> (ถ้าใส่ --fold) หรือ split_signer"""
    if a.fold is not None:
        split_dir = Path(a.split_dir or ROOT / "data" / f"split_fold{a.fold}")
        if not split_dir.exists():
            raise SystemExit(f"ไม่พบ {split_dir} — สร้างก่อนด้วย python ../tools/make_csv.py "
                             f"--split-json ../splits/tsl_one_s_signer_indep_fold{a.fold}.json")
    else:
        split_dir = Path(a.split_dir or SPLIT_DIR)
    a.train_csv = a.train_csv or str(split_dir / "tsl_train.csv")
    a.val_csv = a.val_csv or str(split_dir / "tsl_val.csv")
    a.test_csv = a.test_csv or str(split_dir / "tsl_test.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="ตรวจว่า batched forward = โมเดลเดิม")
    ap.add_argument("--exp", default="b16_test")
    ap.add_argument("--device", choices=["dml", "cpu", "cuda", "mps"], default="dml")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--aug", action="store_true", help="เปิด augmentation ของ SPOTER")
    ap.add_argument("--noise_std", type=float, default=0.001, help="gaussian noise ตาม SPOTER (0 = ปิด)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_classes", type=int, default=184)
    ap.add_argument("--hidden_dim", type=int, default=108)
    ap.add_argument("--pe", choices=["none", "learned", "sin"], default="none",
                    help="positional encoding ตามเวลา (A2): none = SPOTER เดิม, learned / sin = รู้ลำดับเฟรม")
    ap.add_argument("--fold", type=int, help="fold ของทีม: ใช้ ../data/split_fold<k>/ และเขียนผลที่ ../results/")
    ap.add_argument("--split_dir", help=f"โฟลเดอร์ CSV (default {SPLIT_DIR} หรือ ../data/split_fold<k>)")
    ap.add_argument("--select", choices=["top1", "f1"], default="top1",
                    help="เลือก checkpoint จาก val top-1 (เดิม) หรือ val macro-F1 (ตาม brief)")
    ap.add_argument("--patience", type=int, default=0, help="early stopping (0 = ปิด)")
    ap.add_argument("--train_csv")
    ap.add_argument("--val_csv")
    ap.add_argument("--test_csv")
    ap.add_argument("--label_map", default="../data/label_map.json")
    a = ap.parse_args()
    resolve_csv_paths(a)
    check(a) if a.check else train(a)


if __name__ == "__main__":
    main()
