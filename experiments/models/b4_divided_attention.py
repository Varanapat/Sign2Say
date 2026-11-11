"""B4 (L2 — Spatiotemporal, ไม่มี graph prior): Space-time Transformer แบบ divided attention

README_brief §2 Part B: token = 1 จุด × 1 ช่วงเวลา -> ทุกชั้นทำ attention ตามเวลา (จุดเดียวกันข้ามเวลา)
แล้วตามข้อต่อ (ช่วงเวลาเดียวกัน) แบบ TimeSformer — ไม่ทำ full attention เพราะหนักเกิน

ใช้แยกตัวแปรกวนของ B3 (ST-GCN): B4 เป็น L2 เหมือน B3 แต่**ไม่รู้โครงกระดูก**
(ทุกข้อต่อ attend หากันได้หมด ตำแหน่งข้อต่อรู้ผ่าน learnable joint embedding เท่านั้น)

เพื่อให้รันได้บน MPS: รวม 4 เฟรมติดกันเป็น 1 token (temporal patch, แบบ tubelet ของ ViViT)
-> 16 ช่วงเวลา × 54 จุด = 864 token ต่อคลิป
"""
import torch
import torch.nn as nn


class DividedBlock(nn.Module):
    def __init__(self, d, heads, ff, dropout):
        super().__init__()
        self.norm_t, self.norm_s, self.norm_f = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn_t = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_s = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # x [B, T, J, d]
        B, T, J, d = x.shape
        h = x.permute(0, 2, 1, 3).reshape(B * J, T, d)      # temporal: จุดเดียวกันข้ามเวลา
        n = self.norm_t(h)
        h = h + self.drop(self.attn_t(n, n, n, need_weights=False)[0])
        h = h.view(B, J, T, d).permute(0, 2, 1, 3).reshape(B * T, J, d)   # spatial: ข้อต่อในช่วงเดียวกัน
        n = self.norm_s(h)
        h = h + self.drop(self.attn_s(n, n, n, need_weights=False)[0])
        h = h + self.drop(self.ff(self.norm_f(h)))
        return h.view(B, T, J, d)


class DividedSpaceTimeTransformer(nn.Module):
    def __init__(self, input_shape, num_classes, d=160, heads=4, layers=6, ff=640, patch_t=4, dropout=0.1):
        super().__init__()
        T, J, C = input_shape
        if T % patch_t:
            raise ValueError(f"T={T} หาร patch_t={patch_t} ไม่ลงตัว")
        self.patch_t = patch_t
        self.embed = nn.Linear(patch_t * C, d)                       # 4 เฟรม × C channel ของจุดเดียว -> token
        self.time_pos = nn.Parameter(torch.zeros(1, T // patch_t, 1, d))
        self.joint_pos = nn.Parameter(torch.zeros(1, 1, J, d))
        nn.init.trunc_normal_(self.time_pos, std=0.02)
        nn.init.trunc_normal_(self.joint_pos, std=0.02)
        self.blocks = nn.ModuleList([DividedBlock(d, heads, ff, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):                                            # x [B, T, J, C]
        B, T, J, C = x.shape
        p = self.patch_t
        x = x.view(B, T // p, p, J, C).permute(0, 1, 3, 2, 4).reshape(B, T // p, J, p * C)
        h = self.embed(x) + self.time_pos + self.joint_pos           # [B, T/p, J, d]
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h.mean(dim=(1, 2))))
