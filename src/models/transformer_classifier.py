from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.pooling import TrialPooler
from src.models.positional_embeddings import PositionalEmbedding
from src.models.tokenizers import (
    CNNTokenizer,
    FrequencyTokenizer,
    LinearTokenizer,
    PatchPoolTokenizer,
    PatchTokenTokenizer,
)


def validate_tokenizer_pe_combo(tokenizer_type: str, pe_type: str) -> None:
    spatial_tok = {"linear", "cnn", "frequency", "patch_pooling"}
    st_tok = {"patch_token"}
    spatial_pe = {"none", "1d", "2d", "3d"}
    st_pe = {"none", "1d_t", "2d_t", "3d_t"}
    if tokenizer_type in spatial_tok and pe_type not in spatial_pe:
        raise ValueError(f"tokenizer {tokenizer_type} requires spatial PE, got {pe_type}")
    if tokenizer_type in st_tok and pe_type not in st_pe:
        raise ValueError(f"tokenizer {tokenizer_type} requires spatio-temporal PE, got {pe_type}")


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        tokenizer_type: str,
        pe_type: str,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        num_classes: int,
        time_len: int,
        sfreq: float,
        patch_size: int,
        num_channels: int,
        time_encoding_type: str,
        pe_1dt_combine: str,
        pooling_type: str,
        patch_pool_mode: str,
        attention_pooling_heads: int,
        xy: torch.Tensor | None = None,
        xyz: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        validate_tokenizer_pe_combo(tokenizer_type, pe_type)
        self.tokenizer_type = tokenizer_type
        self.pe_type = pe_type
        self.d_model = d_model
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.time_len = time_len

        if tokenizer_type == "linear":
            self.tokenizer = LinearTokenizer(time_len, d_model)
        elif tokenizer_type == "cnn":
            self.tokenizer = CNNTokenizer(d_model)
        elif tokenizer_type == "frequency":
            self.tokenizer = FrequencyTokenizer(d_model, sfreq)
        elif tokenizer_type == "patch_pooling":
            if time_len % patch_size != 0:
                raise ValueError(f"time_len {time_len} must be divisible by patch_size {patch_size}")
            self.tokenizer = PatchPoolTokenizer(
                patch_size, d_model, pool=patch_pool_mode, attention_heads=attention_pooling_heads
            )
        elif tokenizer_type == "patch_token":
            if time_len % patch_size != 0:
                raise ValueError(f"time_len {time_len} must be divisible by patch_size {patch_size}")
            self.tokenizer = PatchTokenTokenizer(patch_size, d_model)
        else:
            raise ValueError(tokenizer_type)

        pe_num_patches = time_len // patch_size if tokenizer_type == "patch_token" else None
        self.pe = PositionalEmbedding(
            pe_type,
            d_model,
            num_channels,
            xy=xy,
            xyz=xyz,
            num_patches=pe_num_patches,
            time_encoding_type=time_encoding_type,
            combine_1dt=pe_1dt_combine,
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pooler = TrialPooler(pooling_type, d_model, attention_pooling_heads)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"expected [B,C,T], got {tuple(x.shape)}")
        b, c, t = x.shape
        if c != self.num_channels:
            raise ValueError(f"expected C={self.num_channels}, got {c}")
        if t != self.time_len:
            raise ValueError(f"expected T={self.time_len}, got {t}")

        tok = self.tokenizer(x)
        if self.tokenizer_type == "patch_token":
            n = self.num_channels * (t // self.patch_size)
        else:
            n = self.num_channels
        if tok.shape[1] != n:
            raise ValueError(f"tokenizer output tokens {tok.shape[1]} != expected {n}")

        pe = self.pe(n, x.device, x.dtype)
        if pe.shape[1] != n:
            raise ValueError(f"PE length {pe.shape[1]} != {n}")
        tok = tok + pe
        tok = self.pooler.prepend_cls(tok)
        tok = self.encoder(tok)
        z = self.pooler.pool(tok)
        return self.head(z)


def build_model(cfg: dict[str, Any], coord_tensors: dict[str, torch.Tensor | None]) -> TransformerClassifier:
    xy = coord_tensors.get("xy")
    xyz = coord_tensors.get("xyz")
    time_len = int(round(float(cfg["epoch_duration"]) * float(cfg["sfreq"])))
    return TransformerClassifier(
        tokenizer_type=cfg["tokenizer_type"],
        pe_type=cfg["pe_type"],
        d_model=int(cfg["d_model"]),
        nhead=int(cfg["nhead"]),
        num_layers=int(cfg["num_layers"]),
        dim_feedforward=int(cfg["dim_feedforward"]),
        dropout=float(cfg["dropout"]),
        num_classes=int(cfg["num_classes"]),
        time_len=time_len,
        sfreq=float(cfg["sfreq"]),
        patch_size=int(cfg["patch_size"]),
        num_channels=int(cfg.get("num_channels", 64)),
        time_encoding_type=cfg["time_encoding_type"],
        pe_1dt_combine=cfg.get("pe_1dt_combine", "add"),
        pooling_type=cfg["pooling_type"],
        patch_pool_mode=cfg.get("patch_pool_mode", "mean"),
        attention_pooling_heads=int(cfg.get("attention_pooling_num_heads", 4)),
        xy=xy,
        xyz=xyz,
    )


def run_dummy_forward_checks(device: torch.device | str = "cpu") -> None:
    dev = torch.device(device)
    time_len = 640
    c = 64
    b = 2
    x = torch.randn(b, c, time_len, device=dev)
    xy = torch.randn(c, 2, device=dev)
    xyz = torch.randn(c, 3, device=dev)

    m1 = TransformerClassifier(
        tokenizer_type="linear",
        pe_type="3d",
        d_model=128,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        num_classes=2,
        time_len=time_len,
        sfreq=160.0,
        patch_size=40,
        num_channels=c,
        time_encoding_type="normalized_linear",
        pe_1dt_combine="add",
        pooling_type="mean",
        patch_pool_mode="mean",
        attention_pooling_heads=4,
        xy=xy,
        xyz=xyz,
    ).to(dev)
    y1 = m1(x)
    assert y1.shape == (b, 2), y1.shape

    m2 = TransformerClassifier(
        tokenizer_type="patch_token",
        pe_type="3d_t",
        d_model=128,
        nhead=4,
        num_layers=1,
        dim_feedforward=256,
        dropout=0.0,
        num_classes=2,
        time_len=time_len,
        sfreq=160.0,
        patch_size=40,
        num_channels=c,
        time_encoding_type="normalized_linear",
        pe_1dt_combine="add",
        pooling_type="mean",
        patch_pool_mode="mean",
        attention_pooling_heads=4,
        xy=xy,
        xyz=xyz,
    ).to(dev)
    y2 = m2(x)
    assert y2.shape == (b, 2), y2.shape
