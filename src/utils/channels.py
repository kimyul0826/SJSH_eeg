from functools import lru_cache

_LEADING_FP = {"FP", "FT", "FC", "CP", "PO", "AF", "TP"}


@lru_cache(maxsize=4096)
def normalize_channel_name(name: str) -> str:
    s = name.strip().replace(".", "").replace(" ", "").replace("-", "")
    if not s:
        return s
    upper = s.upper()

    if upper.endswith("Z") and len(upper) > 1:
        base = upper[:-1]
        last = "z"
    else:
        base = upper
        last = ""

    if len(base) >= 2 and base[:2] in _LEADING_FP:
        prefix = base[:2].title().lower()
        rest = base[2:]
    elif len(base) >= 1 and base[0] in ("F", "C", "P", "T", "O", "A", "I", "D"):
        prefix = base[0]
        rest = base[1:]
    else:
        prefix = base[0] if base else ""
        rest = base[1:] if len(base) > 1 else ""

    rest_norm = rest.upper() if not rest.isdigit() else rest

    out = prefix + rest_norm + last

    fixes = {
        "FPZ": "Fpz",
        "FP1": "Fp1",
        "FP2": "Fp2",
        "FCZ": "FCz",
        "CPZ": "CPz",
        "POZ": "POz",
        "AFZ": "AFz",
        "FZ": "Fz",
        "CZ": "Cz",
        "PZ": "Pz",
        "OZ": "Oz",
        "IZ": "Iz",
        "FT7": "FT7",
        "FT8": "FT8",
        "TP7": "TP7",
        "TP8": "TP8",
        "PO5": "PO5",
        "PO6": "PO6",
        "T9": "T9",
        "T10": "T10",
    }
    upper_key = out.upper()
    if upper_key in fixes:
        return fixes[upper_key]
    return out


def assert_same_channel_order(names_a: list[str], names_b: list[str]) -> None:
    na = [normalize_channel_name(n) for n in names_a]
    nb = [normalize_channel_name(n) for n in names_b]
    if na != nb:
        raise ValueError(f"Channel order mismatch:\n{na}\nvs\n{nb}")


def reorder_indices(raw_names: list[str], canonical: list[str]) -> list[int]:
    norm_raw = [normalize_channel_name(n) for n in raw_names]
    index_map = {n: i for i, n in enumerate(norm_raw)}
    indices = []
    for ch in canonical:
        ch_norm = normalize_channel_name(ch)
        if ch_norm not in index_map:
            raise KeyError(
                f"Canonical channel {ch} (normalized={ch_norm}) not found in recording; "
                f"available sample: {norm_raw[:5]}..."
            )
        indices.append(index_map[ch_norm])
    return indices
