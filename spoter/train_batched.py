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
"""
import argparse
import csv
import random
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
    out = model.transformer(model.pos + h, tgt)                  # (1, B, d)
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


# ---------------------------------------------------------------- check
def check(a):
    torch.manual_seed(0)
    model = SPOTER(num_classes=a.num_classes, hidden_dim=a.hidden_dim).eval()
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
          f"batch {a.batch_size} | frames {a.frames} | aug {a.aug} | device {a.device}")

    model = SPOTER(num_classes=a.num_classes, hidden_dim=a.hidden_dim).to(device)
    if a.device == "dml":
        from dml_adam import DmlAdam
        opt = DmlAdam(model.parameters(), lr=a.lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    crit = torch.nn.CrossEntropyLoss()

    out_dir = Path("out-batched", a.exp)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc", "val_acc", "seconds"])

    best_val, best_ep, t_all = -1.0, 0, time.time()
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
        dt = time.time() - t0
        mark = ""
        if val_acc > best_val:
            best_val, best_ep = val_acc, ep
            # ย้ายไป CPU ก่อนบันทึก: tensor บน DirectML โหลดกลับด้วย weights_only=True ไม่ได้
            torch.save({k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                       out_dir / "best_val.pth")
            mark = "  *best"
        print(f"[{ep:3d}] loss {tot / n:.4f}  train {correct / n:.2%}  val {val_acc:.2%}  ({dt:.0f}s){mark}",
              flush=True)
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tot / n:.5f}", f"{correct / n:.5f}", f"{val_acc:.5f}", f"{dt:.1f}"])

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
    print(f"\nบันทึกผลไว้ที่ {out_dir}")


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
    ap.add_argument("--train_csv", default=SPLIT_DIR + "/tsl_train.csv")
    ap.add_argument("--val_csv", default=SPLIT_DIR + "/tsl_val.csv")
    ap.add_argument("--test_csv", default=SPLIT_DIR + "/tsl_test.csv")
    ap.add_argument("--label_map", default="../data/label_map.json")
    a = ap.parse_args()
    check(a) if a.check else train(a)


if __name__ == "__main__":
    main()
