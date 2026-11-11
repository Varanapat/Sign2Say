"""B0 (L0 — Spatial only): ฝังแต่ละเฟรมด้วย MLP แล้วเฉลี่ยทุกเฟรม -> ไม่รู้ลำดับเวลาเลย (ขอบล่าง)

README_brief §2 Part B: MLP 2 ชั้น (N·C -> 256 -> 256) -> เฉลี่ยตามเวลา -> Linear 184
"""
import torch.nn as nn


class FrameMLPMeanPool(nn.Module):
    def __init__(self, input_shape, num_classes, hidden=256, dropout=0.3):
        super().__init__()
        T, J, C = input_shape
        self.frame = nn.Sequential(
            nn.Linear(J * C, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, num_classes))

    def forward(self, x):                    # x [B, T, J, C]
        h = self.frame(x.flatten(2))         # [B, T, hidden]  ทีละเฟรม
        return self.head(h.mean(dim=1))      # เฉลี่ยตามเวลา -> [B, num_classes]
