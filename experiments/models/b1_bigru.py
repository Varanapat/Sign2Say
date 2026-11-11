"""B1 (L1 — Spatial -> Temporal): ฝังแต่ละเฟรม -> BiGRU ตามเวลา

README_brief §2 Part B (แบบเดียวกับของทีม): Linear 256 -> BiGRU 2 ชั้น hidden 256 -> mean pool -> Linear (~2.1M)
"""
import torch.nn as nn


class FrameBiGRU(nn.Module):
    def __init__(self, input_shape, num_classes, embed=256, hidden=256, layers=2, dropout=0.3):
        super().__init__()
        T, J, C = input_shape
        self.embed = nn.Sequential(nn.Linear(J * C, embed), nn.ReLU(), nn.Dropout(dropout))
        self.gru = nn.GRU(embed, hidden, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * hidden, num_classes))

    def forward(self, x):                      # x [B, T, J, C]
        h = self.embed(x.flatten(2))           # [B, T, embed]   spatial: ทีละเฟรม
        h, _ = self.gru(h)                     # [B, T, 2*hidden] temporal
        return self.head(h.mean(dim=1))
