from functools import lru_cache
from pathlib import Path

import mne
import numpy as np

from src.utils.channels import normalize_channel_name


@lru_cache(maxsize=1)
def _standard_positions() -> dict[str, np.ndarray]:
    montage = mne.channels.make_standard_montage("standard_1005")
    ch_pos = montage.get_positions()["ch_pos"]
    if ch_pos is None:
        raise RuntimeError("Montage has no channel positions")
    out = {}
    for name, pos in ch_pos.items():
        out[normalize_channel_name(name)] = np.asarray(pos, dtype=np.float64)
    return out


def channel_positions_from_names(
    ch_names: list[str],
    canonical: list[str] | None = None,
) -> tuple[list[str], np.ndarray]:
    pos_map = _standard_positions()
    order = [normalize_channel_name(c) for c in (canonical if canonical is not None else ch_names)]
    missing = [ch for ch in order if ch not in pos_map]
    if missing:
        raise KeyError(
            "No standard_1005 position for channel(s): "
            + ", ".join(missing[:10])
            + (" ..." if len(missing) > 10 else "")
        )
    arr = np.stack([pos_map[ch] for ch in order], axis=0)
    return order, arr


def _load_loc_positions(loc_path: str | Path) -> dict[str, np.ndarray]:
    """
    Parse polar .loc format:
      index  theta(deg)  radius  NAME
    Returns NAME->xyz with z=0, x=r*cos(theta), y=r*sin(theta).
    """
    path = Path(loc_path)
    if not path.is_file():
        raise FileNotFoundError(f"loc file not found: {path}")
    out: dict[str, np.ndarray] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        theta_deg = float(parts[1])
        radius = float(parts[2])
        name = normalize_channel_name(parts[-1])
        ang = np.deg2rad(theta_deg)
        out[name] = np.array([radius * np.cos(ang), radius * np.sin(ang), 0.0], dtype=np.float64)
    if not out:
        raise ValueError(f"failed to parse loc positions from {path}")
    return out


def normalize_coords_zero_mean_unit_std(coords: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = coords.mean(axis=0, keepdims=True)
    std = coords.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (coords - mean) / std, mean.squeeze(0), std.squeeze(0)


def save_coord_cache(
    path: str | Path,
    ch_names: list[str],
    xyz: np.ndarray,
    xy_norm: np.ndarray,
    xyz_norm: np.ndarray,
    mean2: np.ndarray,
    std2: np.ndarray,
    mean3: np.ndarray,
    std3: np.ndarray,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        ch_names=np.array(ch_names, dtype=object),
        xyz=xyz,
        xy_norm=xy_norm,
        xyz_norm=xyz_norm,
        mean2=mean2,
        std2=std2,
        mean3=mean3,
        std3=std3,
    )


def load_coord_cache(path: str | Path) -> dict:
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def build_or_load_coordinates(
    ch_names: list[str],
    canonical: list[str] | None,
    cache_path: str | Path | None,
    coord_loc_path: str | Path | None = None,
) -> dict:
    expected_order = [normalize_channel_name(c) for c in (canonical if canonical is not None else ch_names)]
    if cache_path:
        loaded = try_load_coord_cache(cache_path)
        if loaded is not None and loaded["ch_names"] == expected_order:
            return loaded

    try:
        order, xyz = channel_positions_from_names(ch_names, canonical)
    except KeyError:
        if coord_loc_path is None:
            raise
        loc_pos = _load_loc_positions(coord_loc_path)
        order = expected_order
        missing = [ch for ch in order if ch not in loc_pos]
        if missing:
            raise KeyError(
                "Missing loc positions for channel(s): "
                + ", ".join(missing[:10])
                + (" ..." if len(missing) > 10 else "")
            )
        xyz = np.stack([loc_pos[ch] for ch in order], axis=0)
    xy = xyz[:, :2].copy()
    xyz_copy = xyz.copy()
    xy_norm, mean2, std2 = normalize_coords_zero_mean_unit_std(xy)
    xyz_norm, mean3, std3 = normalize_coords_zero_mean_unit_std(xyz_copy)
    meta = {
        "ch_names": order,
        "xyz": xyz_copy,
        "xy_norm": xy_norm.astype(np.float32),
        "xyz_norm": xyz_norm.astype(np.float32),
        "mean2": mean2,
        "std2": std2,
        "mean3": mean3,
        "std3": std3,
    }
    if cache_path:
        save_coord_cache(
            cache_path,
            order,
            xyz_copy,
            meta["xy_norm"],
            meta["xyz_norm"],
            mean2,
            std2,
            mean3,
            std3,
        )
    return meta


def try_load_coord_cache(cache_path: str | Path) -> dict | None:
    path = Path(cache_path)
    if not path.is_file():
        return None
    raw = load_coord_cache(path)
    return {
        "ch_names": [str(x) for x in raw["ch_names"].tolist()],
        "xyz": raw["xyz"],
        "xy_norm": raw["xy_norm"].astype(np.float32),
        "xyz_norm": raw["xyz_norm"].astype(np.float32),
        "mean2": raw["mean2"],
        "std2": raw["std2"],
        "mean3": raw["mean3"],
        "std3": raw["std3"],
    }
