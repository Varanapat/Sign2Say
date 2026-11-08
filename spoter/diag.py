"""
วินิจฉัยว่าทำไม SPOTER ไม่เรียนรู้  (วางไฟล์นี้ในโฟลเดอร์ spoter ที่เดียวกับ train.py แล้วรันจากโฟลเดอร์นั้น)

ส่วนที่ 1: ตรวจข้อมูลที่ loader ของ SPOTER ส่งเข้าโมเดลจริง
    python diag.py data

ส่วนที่ 2: overfit test — ให้โมเดลจำ 32 คลิปเดิมซ้ำๆ  ถ้า pipeline ถูกต้อง accuracy บน 32 คลิปนี้ต้องขึ้นไปใกล้ 100%
    python diag.py overfit --device cpu --opt torch
    python diag.py overfit --device dml --opt torch
    python diag.py overfit --device dml --opt dml      (ต้องมี dml_adam.py ในโฟลเดอร์เดียวกัน)
"""
import argparse
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from datasets.czech_slr_dataset import CzechSLRDataset
from spoter.spoter_model import SPOTER

DEFAULT_CSV = "../data/split_signer/tsl_train.csv"   # ใช้ / ได้ทั้ง Windows และ macOS


def check_data(csv):
    print(f"โหลด {csv} ...")
    ds = CzechSLRDataset(csv)
    print(f"จำนวนคลิป: {len(ds)}")

    labels = [int(ds[i][1].flatten()[0]) for i in range(0, len(ds), max(1, len(ds) // 300))]
    print(f"label ที่สุ่มดู: min={min(labels)}, max={max(labels)}, จำนวนค่าไม่ซ้ำ={len(set(labels))}")

    nan_clips, const_clips, stds, zero_fracs, ranges = 0, 0, [], [], []
    idxs = random.Random(0).sample(range(len(ds)), min(200, len(ds)))
    for i in idxs:
        x, _ = ds[i]
        x = x.float()
        if torch.isnan(x).any() or torch.isinf(x).any():
            nan_clips += 1
            continue
        t_std = x.std(dim=0).mean().item() if x.shape[0] > 1 else 0.0   # การเปลี่ยนแปลงตามเวลา
        stds.append(t_std)
        if t_std < 1e-4:
            const_clips += 1
        zero_fracs.append((x == -0.5).float().mean().item())             # 0 ก่อน shift -0.5 = จุดที่หาย
        ranges.append((x.min().item(), x.max().item()))

    x0, y0 = ds[idxs[0]]
    print(f"ตัวอย่าง 1 คลิป: shape={tuple(x0.shape)}, label={y0.flatten().tolist()}")
    print(f"คลิปที่มี NaN/Inf: {nan_clips}/{len(idxs)}")
    print(f"คลิปที่แทบไม่เปลี่ยนตามเวลา (std<1e-4): {const_clips}/{len(idxs)}")
    if stds:
        print(f"ค่าเฉลี่ย std ตามเวลา: {np.mean(stds):.4f}")
        print(f"สัดส่วนค่าที่เป็น -0.5 (จุดหาย): {np.mean(zero_fracs):.1%}")
        mn = min(r[0] for r in ranges); mx = max(r[1] for r in ranges)
        print(f"ช่วงค่า input: [{mn:.3f}, {mx:.3f}]")
    print("\nตัวอย่างค่าเฟรมกลาง 6 จุดแรก:\n", x0[x0.shape[0] // 2, :6])


def get_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    if name == "mps":                       # Mac Apple Silicon (M1–M4)
        return torch.device("mps")
    import torch_directml
    return torch_directml.device()


def overfit(csv, device_name, opt_name, lr, epochs, n):
    torch.manual_seed(0); random.seed(0); np.random.seed(0)
    device = get_device(device_name)
    ds = CzechSLRDataset(csv)
    sub = Subset(ds, random.Random(0).sample(range(len(ds)), n))
    loader = DataLoader(sub, shuffle=True)

    model = SPOTER(num_classes=184, hidden_dim=108).to(device)
    if opt_name == "dml":
        from dml_adam import DmlAdam
        opt = DmlAdam(model.parameters(), lr=lr)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = torch.nn.CrossEntropyLoss()

    w0 = next(model.parameters()).detach().to("cpu").clone()
    print(f"overfit {n} คลิป | device={device_name} | optimizer={opt_name} | lr={lr}")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot_loss, correct = 0.0, 0
        for x, y in loader:
            x = x.squeeze(0).to(device)
            y = y.to(device, dtype=torch.long).flatten()[:1]
            opt.zero_grad()
            out = model(x).expand(1, -1, -1)[0]
            loss = crit(out, y)
            loss.backward()
            opt.step()
            tot_loss += loss.item()
            correct += int(out.argmax(-1).item() == y.item())
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            w = next(model.parameters()).detach().to("cpu")
            print(f"  epoch {ep:3d}  loss={tot_loss / n:.3f}  acc={correct / n:.0%}  "
                  f"weight เปลี่ยนไป={float((w - w0).abs().mean()):.2e}  ({time.time() - t0:.0f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["data", "overfit"])
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--device", choices=["cpu", "dml", "cuda", "mps"], default="cpu")
    ap.add_argument("--opt", choices=["torch", "dml"], default="torch")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n", type=int, default=32)
    a = ap.parse_args()
    if a.mode == "data":
        check_data(a.csv)
    else:
        overfit(a.csv, a.device, a.opt, a.lr, a.epochs, a.n)


if __name__ == "__main__":
    main()
