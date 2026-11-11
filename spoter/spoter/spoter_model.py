
import copy
from types import SimpleNamespace
import torch

import torch.nn as nn
from typing import Optional


def _sinusoid(length, dim):
    """sinusoidal positional encoding แบบ "Attention Is All You Need" -> (length, dim)"""
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    i = torch.arange(0, dim, 2, dtype=torch.float32)
    angle = pos / torch.pow(10000.0, i / dim)
    pe = torch.zeros(length, dim)
    pe[:, 0::2] = torch.sin(angle)
    pe[:, 1::2] = torch.cos(angle[:, : dim // 2])
    return pe


def _get_clones(mod, n):
    return nn.ModuleList([copy.deepcopy(mod) for _ in range(n)])


class SPOTERTransformerDecoderLayer(nn.TransformerDecoderLayer):
    """
    Edited TransformerDecoderLayer implementation omitting the redundant self-attention operation as opposed to the
    standard implementation.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation):
        super(SPOTERTransformerDecoderLayer, self).__init__(d_model, nhead, dim_feedforward, dropout, activation)

        del self.self_attn
        self.self_attn = SimpleNamespace(batch_first=False)  # ให้ PyTorch 2.x อ่านค่า batch_first ได้

    def forward(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: Optional[torch.Tensor] = None,
                memory_mask: Optional[torch.Tensor] = None, tgt_key_padding_mask: Optional[torch.Tensor] = None,
                memory_key_padding_mask: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:

        tgt = tgt + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt


class SPOTER(nn.Module):
    """
    Implementation of the SPOTER (Sign POse-based TransformER) architecture for sign language recognition from sequence
    of skeletal data.
    """

    def __init__(self, num_classes, hidden_dim=55, pe="none", max_len=64):
        """
        pe (positional encoding ตามเวลา — การทดลอง A2):
          "none"    : แบบเดิม — self.pos (1,1,d) เวกเตอร์เดียวบวกทุกเฟรม -> โมเดลไม่รู้ลำดับเฟรม
          "learned" : self.pos (max_len,1,d) เรียนได้ต่อตำแหน่ง init แบบเดียวกับเดิม (U[0,1))
                      -> ต่างจากเดิมแค่ "ขึ้นกับตำแหน่ง" อย่างเดียว
          "sin"     : self.pos (1,1,d) เดิม + sinusoidal (max_len,1,d) แบบคงที่ (ไม่เรียน)
        """
        super().__init__()
        if pe not in ("none", "learned", "sin"):
            raise ValueError(f"pe ต้องเป็น none / learned / sin ไม่ใช่ {pe!r}")
        self.pe, self.max_len = pe, max_len

        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim))
        if pe == "learned":
            self.pos = nn.Parameter(torch.rand(max_len, 1, hidden_dim))
        else:
            self.pos = nn.Parameter(torch.cat([self.row_embed[0].unsqueeze(0).repeat(1, 1, 1)], dim=-1).flatten(0, 1).unsqueeze(0))
        if pe == "sin":
            self.register_buffer("time_pe", _sinusoid(max_len, hidden_dim).unsqueeze(1), persistent=False)
        self.class_query = nn.Parameter(torch.rand(1, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, 9, 6, 6)
        self.linear_class = nn.Linear(hidden_dim, num_classes)

        # Deactivate the initial attention decoder mechanism
        custom_decoder_layer = SPOTERTransformerDecoderLayer(self.transformer.d_model, self.transformer.nhead, 2048,
                                                             0.1, "relu")
        self.transformer.decoder.layers = _get_clones(custom_decoder_layer, self.transformer.decoder.num_layers)

    def position(self, T: int) -> torch.Tensor:
        """positional term ที่บวกเข้ากับ h รูป (T, B, d) — คืน (1,1,d) หรือ (T,1,d)"""
        if self.pe == "none":
            return self.pos
        if T > self.max_len:
            raise ValueError(f"คลิปยาว {T} เฟรม เกิน max_len={self.max_len} ของ positional encoding "
                             f"— resample ก่อน (train_batched.py ทำให้เป็น 64 เฟรม)")
        if self.pe == "learned":
            return self.pos[:T]
        return self.pos + self.time_pe[:T]

    def forward(self, inputs):
        h = torch.unsqueeze(inputs.flatten(start_dim=1), 1).float()
        h = self.transformer(self.position(h.shape[0]) + h, self.class_query.unsqueeze(0)).transpose(0, 1)
        res = self.linear_class(h)

        return res


if __name__ == "__main__":
    pass
