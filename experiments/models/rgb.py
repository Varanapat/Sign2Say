"""โมเดล RGB ของ Part C (README_brief §2): ResNet-18 ทั้งตระกูล เพื่อให้เทียบกันได้

input x [B, T=16, 3, 112, 112] -> logits [B, num_classes]
  C1 (L0) ResNet-18 ต่อเฟรม (ImageNet) -> เฉลี่ยทุกเฟรม
  C2 (L1) ResNet-18 ต่อเฟรม (ImageNet) -> BiGRU (CNN-LSTM ตามบท Ch13)
  C3 (L2 แบบแยก conv) R(2+1)D-18 (Kinetics-400)
  C4 (L2 เต็ม)         R3D-18 (Kinetics-400)
pretrained=False ใช้ทดสอบโดยไม่ต้องดาวน์โหลด weight
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.video import R2Plus1D_18_Weights, R3D_18_Weights, r2plus1d_18, r3d_18


class FrameResNet(nn.Module):
    """C1 / C2: ResNet-18 ทีละเฟรม แล้วรวมเวลาด้วย mean (C1) หรือ BiGRU (C2)"""

    def __init__(self, num_classes, temporal="mean", pretrained=True, hidden=256, dropout=0.3):
        super().__init__()
        self.cnn = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.cnn.fc = nn.Identity()                               # -> 512 ต่อเฟรม
        self.temporal = temporal
        if temporal == "gru":
            self.gru = nn.GRU(512, hidden, batch_first=True, bidirectional=True)
            out = 2 * hidden
        else:
            out = 512
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out, num_classes))

    def forward(self, x):                                          # [B, T, 3, H, W]
        B, T = x.shape[:2]
        h = self.cnn(x.flatten(0, 1)).view(B, T, -1)               # spatial ทีละเฟรม
        if self.temporal == "gru":
            h, _ = self.gru(h)                                     # temporal
        return self.head(h.mean(dim=1))


class VideoResNet(nn.Module):
    """C3 / C4: 3D CNN ของ torchvision — conv มองทั้งพื้นที่และเวลาตั้งแต่ชั้นแรก"""

    def __init__(self, num_classes, arch="r3d", pretrained=True, dropout=0.3):
        super().__init__()
        if arch == "r2plus1d":
            self.net = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1 if pretrained else None)
        else:
            self.net = r3d_18(weights=R3D_18_Weights.KINETICS400_V1 if pretrained else None)
        self.net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.net.fc.in_features, num_classes))

    def forward(self, x):                                          # [B, T, 3, H, W]
        return self.net(x.permute(0, 2, 1, 3, 4))                  # -> [B, 3, T, H, W]


# name -> (builder(num_classes, pretrained), level, normalization stats)
RGB_MODELS = {
    "C1": (lambda n, p: FrameResNet(n, "mean", p), "L0", "imagenet"),
    "C2": (lambda n, p: FrameResNet(n, "gru", p), "L1", "imagenet"),
    "C3": (lambda n, p: VideoResNet(n, "r2plus1d", p), "L2", "kinetics"),
    "C4": (lambda n, p: VideoResNet(n, "r3d", p), "L2", "kinetics"),
}
