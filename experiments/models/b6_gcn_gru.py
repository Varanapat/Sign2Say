"""B6 (L1 — Spatial -> Temporal, มี graph prior): GCN ต่อเฟรม -> BiGRU ตามเวลา

ใช้แยกตัวแปรกวนของ B3 (ST-GCN): ใช้กราฟโครงกระดูกและบล็อก graph conv ชุดเดียวกับ B3
แต่ **temporal kernel = 1** (แต่ละเฟรมแยกกันในส่วน spatial) แล้วค่อยเรียนเวลาด้วย BiGRU แบบเดียวกับ B1

  B6 vs B1 -> ผลของ graph prior (ทั้งคู่เป็น L1)
  B3 vs B6 -> ผลของการเรียนพื้นที่ + เวลา "พร้อมกัน" (ทั้งคู่มี graph prior)
"""
import torch.nn as nn

from .b3_stgcn import STGCNBlock, c54_edges, spatial_partition


class GCNThenBiGRU(nn.Module):
    CHANNELS = [64, 64, 128, 128, 128, 128]   # จบที่ 128: ขนาดรวมใกล้ B1 (2.1M) และเร็วพอบน MPS

    def __init__(self, input_shape, num_classes, hidden=256, gru_layers=2, dropout=0.3):
        super().__init__()
        T, V, C = input_shape
        if V != 54:
            raise ValueError("กราฟนิยามไว้สำหรับ C54 (54 จุด) เท่านั้น")
        A = spatial_partition(V, c54_edges())
        self.data_bn = nn.BatchNorm1d(V * C)
        blocks, c_prev = [], C
        for i, c in enumerate(self.CHANNELS):          # t_kernel=1, stride 1 -> ไม่ผสมข้อมูลข้ามเฟรม
            blocks.append(STGCNBlock(c_prev, c, A, stride=1, t_kernel=1, dropout=0.1, residual=i > 0))
            c_prev = c
        self.gcn = nn.Sequential(*blocks)
        self.gru = nn.GRU(c_prev, hidden, num_layers=gru_layers, batch_first=True,
                          bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x):                              # x [B, T, V, C]
        B, T, V, C = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(B, V * C, T)).view(B, V, C, T).permute(0, 2, 3, 1)
        h = self.gcn(x)                                # [B, C', T, V]  spatial ต่อเฟรม
        h = h.mean(dim=3).transpose(1, 2)              # เฉลี่ยข้อต่อ -> [B, T, C']
        h, _ = self.gru(h)                             # temporal
        return self.head(h.mean(dim=1))
