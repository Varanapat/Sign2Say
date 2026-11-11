"""Hand-local features and ST-GCN for C54, used to bring the friend's RQ-S findings into our protocol.

Re-implemented from the description in the friend's EXPERIMENTS.md / B3 (Sign2Say.zip, 11 Nov 2026):
  hand_local : per frame, each hand's xy relative to its wrist, divided by the hand's size
               (longer side of its bounding box); missing points and non-hand points = 0.
               Computed in torch inside the model, so it follows our online augmentation.
  STGCN      : Yan et al. 2018. 9 blocks (64, 64, 64, 128/s2, 128, 128, 256/s2, 256, 256),
               spatial-partition adjacency (self / towards / away from the shoulders) with learned
               edge importance, temporal conv k = 9, data BatchNorm per joint and channel,
               global average pool -> Linear.
C54 order: 0 l_shoulder 1 r_shoulder 2 l_elbow 3 r_elbow 4 l_wrist 5 r_wrist,
           6-9 brows, 10 mouth_right 11 mouth_left, 12-32 left hand, 33-53 right hand.
"""

import numpy as np
import torch
import torch.nn as nn

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10),
              (10, 11), (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18),
              (18, 19), (19, 20), (0, 17)]
LH, RH = 12, 33


def hand_local(x):
    """x [B, T, 54, C>=3] (xyz first, missing points = 0) -> [B, T, 54, 2]."""
    out = torch.zeros(*x.shape[:3], 2, device=x.device, dtype=x.dtype)
    for s in (LH, RH):
        h = x[:, :, s:s + 21, :2]                                     # [B,T,21,2]
        v = (x[:, :, s:s + 21, :3] != 0).any(-1)                      # [B,T,21]
        vf = v.unsqueeze(-1).to(x.dtype)
        n = vf.sum(2, keepdim=True).clamp(min=1)
        mean = (h * vf).sum(2, keepdim=True) / n
        origin = torch.where(v[:, :, :1, None], h[:, :, :1], mean)    # wrist, else mean of valid
        big = torch.finfo(x.dtype).max / 4
        hi = torch.where(v[..., None], h, torch.full_like(h, -big)).amax(2)
        lo = torch.where(v[..., None], h, torch.full_like(h, big)).amin(2)
        size = (hi - lo).amax(-1)                                     # [B,T]
        size = torch.where(v.any(2) & (size > 1e-3), size, torch.ones_like(size))
        out[:, :, s:s + 21] = (h - origin) / size[..., None, None] * vf
    return out


class WithHandLocal(nn.Module):
    """Appends the 2 hand-local channels to the 7 input channels, then calls the inner model."""
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(torch.cat([x, hand_local(x)], -1))


def c54_edges():
    e = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (4, LH), (5, RH),
         (6, 7), (7, 8), (8, 9), (10, 11), (7, 11), (8, 10), (11, 0), (10, 1)]
    e += [(LH + a, LH + b) for a, b in HAND_EDGES]
    e += [(RH + a, RH + b) for a, b in HAND_EDGES]
    return e


def spatial_partition(V=54, edges=None, centers=(0, 1)):
    edges = edges or c54_edges()
    adj = np.zeros((V, V))
    for a, b in edges:
        adj[a, b] = adj[b, a] = 1
    hop = np.full(V, np.inf)
    hop[list(centers)] = 0
    frontier = list(centers)
    while frontier:
        nxt = []
        for u in frontier:
            for w in np.flatnonzero(adj[u]):
                if hop[w] == np.inf:
                    hop[w] = hop[u] + 1
                    nxt.append(w)
        frontier = nxt
    assert np.isfinite(hop).all(), "C54 graph is not connected"
    A = np.zeros((3, V, V))
    A[0] = np.eye(V)
    for i in range(V):
        for j in np.flatnonzero(adj[i]):
            A[1 if hop[j] < hop[i] else 2 if hop[j] > hop[i] else 0, i, j] = 1
    A = A / A.sum(axis=(0, 2), keepdims=True).clip(min=1)
    return torch.tensor(A, dtype=torch.float32)


class STGCNBlock(nn.Module):
    def __init__(self, c_in, c_out, A, stride=1, k=9, dropout=0.1, residual=True):
        super().__init__()
        self.K = A.shape[0]
        self.register_buffer("A", A)
        self.importance = nn.Parameter(torch.ones_like(A))
        self.gcn = nn.Conv2d(c_in, c_out * self.K, 1)
        self.tcn = nn.Sequential(nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                                 nn.Conv2d(c_out, c_out, (k, 1), (stride, 1), ((k - 1) // 2, 0)),
                                 nn.BatchNorm2d(c_out), nn.Dropout(dropout))
        if not residual:
            self.res = None
        elif c_in == c_out and stride == 1:
            self.res = nn.Identity()
        else:
            self.res = nn.Sequential(nn.Conv2d(c_in, c_out, 1, (stride, 1)), nn.BatchNorm2d(c_out))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):                                    # [B,C,T,V]
        B, _, T, V = x.shape
        h = self.gcn(x).view(B, self.K, -1, T, V)
        h = torch.einsum("bkctv,kvw->bctw", h, self.A * self.importance)
        h = self.tcn(h)
        if self.res is not None:
            h = h + self.res(x)
        return self.relu(h)


class STGCN(nn.Module):
    CHANNELS = [(64, 1), (64, 1), (64, 1), (128, 2), (128, 1), (128, 1), (256, 2), (256, 1), (256, 1)]

    def __init__(self, n_classes, C=7, V=54, dropout=0.1):
        super().__init__()
        A = spatial_partition(V)
        self.data_bn = nn.BatchNorm1d(V * C)
        blocks, c_prev = [], C
        for i, (c, s) in enumerate(self.CHANNELS):
            blocks.append(STGCNBlock(c_prev, c, A, s, dropout=dropout, residual=i > 0))
            c_prev = c
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c_prev, n_classes))

    def forward(self, x):                                    # [B,T,V,C]
        B, T, V, C = x.shape
        x = self.data_bn(x.permute(0, 2, 3, 1).reshape(B, V * C, T))
        x = x.view(B, V, C, T).permute(0, 2, 3, 1)
        return self.head(self.blocks(x).mean(dim=(2, 3)))
