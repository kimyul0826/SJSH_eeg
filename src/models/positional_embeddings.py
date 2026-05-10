from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def time_coord_values(num_patches: int, time_encoding_type: str) -> np.ndarray:
    if time_encoding_type == "normalized_linear":
        return np.linspace(-1.0, 1.0, num_patches, dtype=np.float32)
    if time_encoding_type == "normalized_0_1":
        return np.linspace(0.0, 1.0, num_patches, dtype=np.float32)
    if time_encoding_type == "index":
        return np.arange(num_patches, dtype=np.float32)
    raise ValueError(time_encoding_type)


class CoordMLP(nn.Module):
    def __init__(self, in_dim: int, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        pe_type: str,
        d_model: int,
        num_channels: int,
        *,
        xy: torch.Tensor | None = None,
        xyz: torch.Tensor | None = None,
        num_patches: int | None = None,
        time_encoding_type: str = "normalized_linear",
        combine_1dt: str = "add",
    ) -> None:
        super().__init__()
        self.pe_type = pe_type
        self.d_model = d_model
        self.num_channels = num_channels
        self.combine_1dt = combine_1dt
        self.time_encoding_type = time_encoding_type

        if xy is not None:
            self.register_buffer("xy", xy.float().clone())
        else:
            self.xy = None
        if xyz is not None:
            self.register_buffer("xyz", xyz.float().clone())
        else:
            self.xyz = None

        if pe_type == "none":
            pass
        elif pe_type == "1d":
            self.ch_emb = nn.Embedding(num_channels, d_model)
        elif pe_type == "2d":
            if self.xy is None:
                raise ValueError("2d PE requires xy")
            self.mlp = CoordMLP(2, d_model)
        elif pe_type == "3d":
            if self.xyz is None:
                raise ValueError("3d PE requires xyz")
            self.mlp = CoordMLP(3, d_model)
        elif pe_type == "1d_t":
            if num_patches is None:
                raise ValueError("1d_t requires num_patches")
            self.ch_emb = nn.Embedding(num_channels, d_model)
            if time_encoding_type == "index":
                self.time_emb = nn.Embedding(num_patches, d_model)
            else:
                tvals = time_coord_values(num_patches, time_encoding_type)
                self.register_buffer("t_values", torch.from_numpy(tvals))
                self.time_proj = nn.Sequential(
                    nn.Linear(1, d_model),
                    nn.GELU(),
                    nn.Linear(d_model, d_model),
                )
            if combine_1dt == "concat":
                self.combine = nn.Linear(2 * d_model, d_model)
        elif pe_type == "2d_t":
            if self.xy is None or num_patches is None:
                raise ValueError("2d_t requires xy and num_patches")
            tvals = time_coord_values(num_patches, time_encoding_type)
            self.register_buffer("t_values", torch.from_numpy(tvals))
            self.mlp = CoordMLP(3, d_model)
        elif pe_type == "3d_t":
            if self.xyz is None or num_patches is None:
                raise ValueError("3d_t requires xyz and num_patches")
            tvals = time_coord_values(num_patches, time_encoding_type)
            self.register_buffer("t_values", torch.from_numpy(tvals))
            self.mlp = CoordMLP(4, d_model)
        else:
            raise ValueError(pe_type)

    def forward(self, num_tokens: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.pe_type == "none":
            return torch.zeros(1, num_tokens, self.d_model, device=device, dtype=dtype)

        if self.pe_type == "1d":
            idx = torch.arange(self.num_channels, device=device)
            return self.ch_emb(idx).unsqueeze(0).to(dtype=dtype)

        if self.pe_type == "2d":
            assert self.xy is not None
            coords = self.xy.to(device=device, dtype=dtype)
            pe = self.mlp(coords)
            return pe.unsqueeze(0).to(dtype=dtype)

        if self.pe_type == "3d":
            assert self.xyz is not None
            coords = self.xyz.to(device=device, dtype=dtype)
            pe = self.mlp(coords)
            return pe.unsqueeze(0).to(dtype=dtype)

        if self.pe_type == "1d_t":
            if self.time_encoding_type == "index":
                p = int(self.time_emb.num_embeddings)
            else:
                p = int(self.t_values.shape[0])
            c = self.num_channels
            assert num_tokens == c * p
            ch = torch.arange(c, device=device)
            ch_e = self.ch_emb(ch).unsqueeze(1).expand(-1, p, -1).reshape(c * p, -1).to(dtype=dtype)
            if self.time_encoding_type == "index":
                assert hasattr(self, "time_emb")
                t_idx = torch.arange(p, device=device)
                te = self.time_emb(t_idx).unsqueeze(0).expand(c, -1, -1).reshape(c * p, -1).to(dtype=dtype)
            else:
                assert hasattr(self, "time_proj") and hasattr(self, "t_values")
                t_raw = self.t_values.to(device=device, dtype=dtype)
                tt = t_raw.view(1, p, 1).expand(c, -1, -1).reshape(c * p, 1)
                te = self.time_proj(tt).to(dtype=dtype)
            if self.combine_1dt == "add":
                pe = ch_e + te
            else:
                pe = self.combine(torch.cat([ch_e, te], dim=-1))
            return pe.unsqueeze(0)

        if self.pe_type in ("2d_t", "3d_t"):
            assert self.t_values is not None
            p = int(self.t_values.shape[0])
            c = self.num_channels
            assert num_tokens == c * p
            t_raw = self.t_values.to(device=device, dtype=dtype).view(1, p, 1).expand(c, -1, -1)
            if self.pe_type == "2d_t":
                assert self.xy is not None
                spatial = self.xy.to(device=device, dtype=dtype).unsqueeze(1).expand(-1, p, -1)
                inp = torch.cat([spatial, t_raw], dim=-1)
            else:
                assert self.xyz is not None
                spatial = self.xyz.to(device=device, dtype=dtype).unsqueeze(1).expand(-1, p, -1)
                inp = torch.cat([spatial, t_raw], dim=-1)
            pe = self.mlp(inp.reshape(c * p, -1))
            return pe.unsqueeze(0)

        raise RuntimeError("unreachable")
