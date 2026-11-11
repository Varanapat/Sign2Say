"""ทะเบียนโมเดลของ Part B / C

เพิ่มโมเดลใหม่: เขียนคลาสที่รับ x [B, T, J, C] -> logits [B, num_classes]
แล้วลงทะเบียนใน MODELS ด้วย (builder, level, modality)
  builder(input_shape=(T, J, C), num_classes) -> nn.Module
"""
from .b0_frame_mlp import FrameMLPMeanPool
from .b1_bigru import FrameBiGRU
from .b2_transformer import FrameTransformer
from .b3_stgcn import STGCN
from .b4_divided_attention import DividedSpaceTimeTransformer
from .b6_gcn_gru import GCNThenBiGRU

MODELS = {
    "B0": (FrameMLPMeanPool, "L0", "landmark"),
    "B1": (FrameBiGRU, "L1", "landmark"),
    "B2": (FrameTransformer, "L1", "landmark"),
    "B3": (STGCN, "L2", "landmark"),
    "B4": (DividedSpaceTimeTransformer, "L2", "landmark"),   # L2 ไม่มี graph prior
    "B6": (GCNThenBiGRU, "L1", "landmark"),                  # L1 มี graph prior
}


def build(name: str, input_shape, num_classes: int):
    if name not in MODELS:
        raise SystemExit(f"ไม่รู้จักโมเดล {name} — มี: {', '.join(MODELS)}")
    builder, level, modality = MODELS[name]
    return builder(input_shape, num_classes), level, modality
