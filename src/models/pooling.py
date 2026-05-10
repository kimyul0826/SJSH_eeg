import torch
import torch.nn as nn


class AttentionPool(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = x.mean(dim=1, keepdim=True)
        out, _ = self.attn(q, x, x, need_weights=False)
        return out.squeeze(1)


class TrialPooler(nn.Module):
    def __init__(self, pooling_type: str, d_model: int, attention_num_heads: int = 4) -> None:
        super().__init__()
        self.pooling_type = pooling_type
        if pooling_type == "attention":
            self.attn_pool = AttentionPool(d_model, attention_num_heads)
        elif pooling_type == "cls":
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        elif pooling_type == "mean":
            pass
        else:
            raise ValueError(f"Unknown pooling_type {pooling_type}")

    def prepend_cls(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_type != "cls":
            return x
        b = x.size(0)
        cls = self.cls.expand(b, -1, -1)
        return torch.cat([cls, x], dim=1)

    def pool(self, x: torch.Tensor) -> torch.Tensor:
        if self.pooling_type == "mean":
            return x.mean(dim=1)
        if self.pooling_type == "cls":
            return x[:, 0, :]
        if self.pooling_type == "attention":
            return self.attn_pool(x)
        raise RuntimeError("unreachable")
