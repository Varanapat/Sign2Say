"""B2 (L1 — Spatial -> Temporal): ฝังแต่ละเฟรม -> Transformer encoder ตามเวลา

README_brief §2 Part B: 4 ชั้น, d = 256, 4 heads, ff 1024, + positional encoding, [CLS] token
ต่างจาก SPOTER: มี positional encoding จริง (learnable ต่อตำแหน่ง) และไม่มี decoder
"""
import torch
import torch.nn as nn


class FrameTransformer(nn.Module):
    def __init__(self, input_shape, num_classes, d=256, heads=4, layers=4, ff=1024, dropout=0.1):
        super().__init__()
        T, J, C = input_shape
        self.embed = nn.Linear(J * C, d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, T + 1, d))          # +1 สำหรับ [CLS]
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, num_classes)

    def forward(self, x):                                           # x [B, T, J, C]
        h = self.embed(x.flatten(2))                                # [B, T, d]
        h = torch.cat([self.cls.expand(len(h), -1, -1), h], dim=1)  # [B, T+1, d]
        h = self.encoder(h + self.pos[:, :h.shape[1]])
        return self.head(self.norm(h[:, 0]))                        # ใช้ [CLS]
