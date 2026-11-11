"""B3 (L2 — Spatiotemporal): ST-GCN (Yan et al., AAAI 2018) สำหรับ C54

ทุกบล็อกมองทั้ง "ข้อต่อข้างเคียง" (graph conv ตามกระดูก) และ "เฟรมข้างเคียง" (temporal conv k=9)
README_brief §2 Part B: 6–9 บล็อก, channel 64 -> 128 -> 256

ลำดับจุดใน C54 (my_work/scripts/landmarks.py):
   0 l_shoulder  1 r_shoulder  2 l_elbow  3 r_elbow  4 l_wrist  5 r_wrist
   6 lbrow_outer 7 lbrow_inner 8 rbrow_inner 9 rbrow_outer 10 mouth_right 11 mouth_left
  12–32 มือซ้าย (MediaPipe hand 0–20)   33–53 มือขวา
"""
import numpy as np
import torch
import torch.nn as nn

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
LH, RH = 12, 33


def c54_edges():
    e = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5),           # ไหล่-ศอก-ข้อมือ
         (4, LH), (5, RH),                                 # ข้อมือ (pose) -> ข้อมือ (hand)
         (6, 7), (7, 8), (8, 9), (10, 11),                 # คิ้ว, มุมปาก
         (7, 11), (8, 10),                                 # คิ้ว -> ปาก (หน้าเป็นกลุ่มเดียวกัน)
         (11, 0), (10, 1)]                                 # ปาก -> ไหล่ (ต่อหน้าเข้ากับลำตัว ไม่มีจุดคอ)
    e += [(LH + a, LH + b) for a, b in HAND_EDGES]
    e += [(RH + a, RH + b) for a, b in HAND_EDGES]
    return e


def spatial_partition(num_nodes, edges, centers=(0, 1)):
    """adjacency 3 ชุดแบบ ST-GCN 'spatial': ตัวเอง / เข้าหาศูนย์กลาง / ออกจากศูนย์กลาง -> [3, N, N]"""
    adj = np.zeros((num_nodes, num_nodes))
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1
    # ระยะ hop จากศูนย์กลาง (ไหล่ทั้งสอง)
    hop = np.full(num_nodes, np.inf)
    frontier = list(centers)
    hop[frontier] = 0
    while frontier:
        nxt = []
        for u in frontier:
            for v in np.flatnonzero(adj[u]):
                if hop[v] == np.inf:
                    hop[v] = hop[u] + 1
                    nxt.append(v)
        frontier = nxt
    if np.isinf(hop).any():
        raise ValueError(f"กราฟไม่เชื่อมกัน: จุด {np.flatnonzero(np.isinf(hop)).tolist()}")

    A = np.zeros((3, num_nodes, num_nodes))
    A[0] = np.eye(num_nodes)
    for i in range(num_nodes):
        for j in np.flatnonzero(adj[i]):        # j เป็นเพื่อนบ้านของ i
            A[1 if hop[j] < hop[i] else 2 if hop[j] > hop[i] else 0, i, j] = 1
    # normalize ตาม degree (แต่ละแถวรวมเป็น 1 ต่อ subset)
    deg = A.sum(axis=(0, 2), keepdims=True).clip(min=1)
    return torch.tensor(A / deg, dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A, stride=1, t_kernel=9, dropout=0.1, residual=True):
        super().__init__()
        K = A.shape[0]
        self.register_buffer("A", A)
        self.edge_importance = nn.Parameter(torch.ones_like(A))      # เรียนน้ำหนักของแต่ละ edge
        self.gcn = nn.Conv2d(c_in, c_out * K, 1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, (t_kernel, 1), (stride, 1), ((t_kernel - 1) // 2, 0)),
            nn.BatchNorm2d(c_out), nn.Dropout(dropout))
        if not residual:
            self.res = None
        elif c_in == c_out and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)
        self.K = K

    def forward(self, x):                                  # x [B, C, T, V]
        B, _, T, V = x.shape
        h = self.gcn(x).view(B, self.K, -1, T, V)          # [B, K, C_out, T, V]
        h = torch.einsum("bkctv,kvw->bctw", h, self.A * self.edge_importance)  # spatial (graph)
        h = self.tcn(h)                                    # temporal
        if self.res is not None:
            h = h + self.res(x)
        return self.relu(h)


class STGCN(nn.Module):
    CHANNELS = [(64, 1), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1)]

    def __init__(self, input_shape, num_classes, dropout=0.1):
        super().__init__()
        T, V, C = input_shape
        if V != 54:
            raise ValueError("กราฟของ ST-GCN นิยามไว้สำหรับ C54 (54 จุด) เท่านั้น")
        A = spatial_partition(V, c54_edges())
        self.data_bn = nn.BatchNorm1d(V * C)
        blocks, c_prev = [], C
        for i, (c, s) in enumerate(self.CHANNELS):
            blocks.append(STGCNBlock(c_prev, c, A, stride=s, dropout=dropout, residual=i > 0))
            c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(c_prev, num_classes)

    def forward(self, x):                                  # x [B, T, V, C]
        B, T, V, C = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(B, V * C, T))       # BN ต่อจุดต่อ channel
        x = x.view(B, V, C, T).permute(0, 2, 3, 1)                          # [B, C, T, V]
        x = self.blocks(x)
        return self.head(x.mean(dim=(2, 3)))                                # global average pool
