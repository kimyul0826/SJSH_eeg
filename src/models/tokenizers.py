import torch
import torch.nn as nn
class LinearTokenizer(nn.Module):
    def __init__(self, time_len: int, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(time_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class CNNTokenizer(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Conv1d(64, d_model, kernel_size=9, padding=4),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = x.reshape(b * c, 1, t)
        y = self.net(y).squeeze(-1)
        return y.view(b, c, -1)


class FrequencyTokenizer(nn.Module):
    def __init__(self, d_model: int, sfreq: float, bands_hz: list[tuple[float, float]] | None = None) -> None:
        super().__init__()
        self.sfreq = sfreq
        self.bands = bands_hz or [(4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 40.0)]
        num_bands = len(self.bands)
        self.mlp = nn.Sequential(
            nn.Linear(num_bands, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self._cached_t: int = -1
        self._cached_device: torch.device | None = None
        self._cached_dtype: torch.dtype | None = None
        self._cached_masks: list[torch.Tensor] = []

    def _get_band_masks(self, t: int, device: torch.device, dtype: torch.dtype) -> list[torch.Tensor]:
        if t == self._cached_t and device == self._cached_device and dtype == self._cached_dtype:
            return self._cached_masks
        freqs = torch.fft.rfftfreq(t, d=1.0 / self.sfreq).to(device=device, dtype=dtype)
        masks = [(freqs >= lo) & (freqs < hi) for (lo, hi) in self.bands]
        self._cached_masks = masks
        self._cached_t = t
        self._cached_device = device
        self._cached_dtype = dtype
        return masks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        spec = torch.fft.rfft(x, dim=-1)
        psd = (spec.abs() ** 2) / float(t**2)
        masks = self._get_band_masks(t, x.device, x.dtype)
        feats = [psd[..., m].sum(dim=-1) for m in masks]
        stacked = torch.stack(feats, dim=-1)
        return self.mlp(stacked)


class PatchPoolTokenizer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        d_model: int,
        pool: str = "mean",
        attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if pool not in ("mean", "attention"):
            raise ValueError(pool)
        self.patch_size = patch_size
        self.pool = pool
        self.proj = nn.Linear(patch_size, d_model)
        if pool == "attention":
            self.attn = nn.MultiheadAttention(d_model, attention_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.patch_size != 0:
            raise ValueError(f"T={t} not divisible by patch_size={self.patch_size}")
        p = t // self.patch_size
        y = x.reshape(b, c, p, self.patch_size)
        y = self.proj(y)
        if self.pool == "mean":
            return y.mean(dim=2)
        flat = y.reshape(b * c, p, -1)
        q = flat.mean(dim=1, keepdim=True)
        out, _ = self.attn(q, flat, flat, need_weights=False)
        return out.squeeze(1).reshape(b, c, -1)


class PatchTokenTokenizer(nn.Module):
    def __init__(self, patch_size: int, d_model: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(patch_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        if t % self.patch_size != 0:
            raise ValueError(f"T={t} not divisible by patch_size={self.patch_size}")
        p = t // self.patch_size
        y = x.reshape(b, c, p, self.patch_size)
        y = self.proj(y)
        return y.reshape(b, c * p, -1)
