from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mne
import numpy as np
import torch
from mne.filter import resample
from torch.utils.data import Dataset

from src.utils.channels import assert_same_channel_order, normalize_channel_name, reorder_indices


def list_subject_ids(data_root: str | Path) -> list[str]:
    root = Path(data_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if p.is_dir() and len(p.name) == 4 and p.name[0] == "S" and p.name[1:].isdigit():
            out.append(p.name[1:])
    return out


def _subject_folder(sid: str) -> str:
    return f"S{int(sid):03d}"


def _edf_path(data_root: Path, sid: str, run: int) -> Path:
    folder = _subject_folder(sid)
    return data_root / folder / f"{folder}R{run:02d}.edf"


def _annotation_labels(event_id: dict[str, int]) -> tuple[int | None, int | None]:
    t1 = t2 = None
    for k, v in event_id.items():
        ku = k.strip().upper()
        if ku == "T1":
            t1 = v
        elif ku == "T2":
            t2 = v
    return t1, t2


@dataclass
class TrialIndex:
    edf_path: str
    onset_sample: int
    label: int
    subject_id: str
    run_id: int


class RawLRU:
    def __init__(self, max_items: int = 6) -> None:
        self.max_items = max_items
        self._store: OrderedDict[str, mne.io.BaseRaw] = OrderedDict()

    def get(self, path: str) -> mne.io.BaseRaw:
        if path in self._store:
            self._store.move_to_end(path)
            return self._store[path]
        raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        self._store[path] = raw
        while len(self._store) > self.max_items:
            self._store.popitem(last=False)
        return raw


class EEGMMIDBDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        subject_ids: list[str],
        runs: list[int],
        epoch_duration: float,
        sfreq: float,
        *,
        exclude_runs: list[int] | None = None,
        canonical_channels: list[str] | None = None,
        verify_channel_order: bool = True,
        resample_to_sfreq: bool = True,
        raw_cache_size: int = 6,
        expected_n_channels: int = 64,
        label_scheme: str = "t1_t2",
    ) -> None:
        super().__init__()
        self.data_root = Path(data_root)
        self.epoch_duration = float(epoch_duration)
        self.target_sfreq = float(sfreq)
        self.canonical_channels = canonical_channels
        self.verify_channel_order = verify_channel_order
        self.resample_to_sfreq = resample_to_sfreq
        self.raw_cache = RawLRU(max_items=raw_cache_size)
        self.expected_n_channels = expected_n_channels
        self.label_scheme = str(label_scheme)
        self._picks_cache: dict[str, np.ndarray] = {}

        exclude = set(exclude_runs or [])
        runs_eff = [r for r in runs if r not in exclude]

        self._indices: list[TrialIndex] = []
        self._reference_ch_names: list[str] | None = None
        self._reorder_idx: list[int] | None = None

        for sid in subject_ids:
            for run in runs_eff:
                path = _edf_path(self.data_root, sid, run)
                if not path.is_file():
                    continue
                self._index_file(str(path), sid, run)

        if not self._indices:
            raise FileNotFoundError(
                f"No trials indexed under {self.data_root}. "
                "Ensure EDF files exist (e.g. S001/S001R04.edf)."
            )

        if self.canonical_channels is not None and self._reference_ch_names is not None:
            self._reorder_idx = reorder_indices(self._reference_ch_names, self.canonical_channels)

    @property
    def reference_channel_names(self) -> list[str]:
        if self._reference_ch_names is None:
            raise RuntimeError("No reference channels (empty dataset?)")
        return list(self._reference_ch_names)

    @property
    def normalized_channel_names(self) -> list[str]:
        return [normalize_channel_name(n) for n in self.reference_channel_names]

    def _index_file(self, path: str, subject_id: str, run_id: int) -> None:
        raw = mne.io.read_raw_edf(path, preload=False, verbose="ERROR")
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        t1_id, t2_id = _annotation_labels(event_id)
        if t1_id is None or t2_id is None:
            return

        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        ch_names = [raw.ch_names[i] for i in picks]
        norm_names = [normalize_channel_name(c) for c in ch_names]

        if self._reference_ch_names is None:
            self._reference_ch_names = list(ch_names)
        elif self.verify_channel_order:
            assert_same_channel_order(self._reference_ch_names, ch_names)

        sfreq = raw.info["sfreq"]
        n_samples = int(round(self.epoch_duration * sfreq))
        for ev in events:
            et = int(ev[2])
            if et == t1_id:
                base_label = 0
            elif et == t2_id:
                base_label = 1
            else:
                continue

            if self.label_scheme == "t1_t2":
                label = base_label
            elif self.label_scheme == "t1t2_by_run_group":
                if int(run_id) in (4, 8, 12):
                    run_group = 0
                elif int(run_id) in (6, 10, 14):
                    run_group = 1
                else:
                    raise ValueError(f"Unknown run_id {run_id} for label_scheme=t1t2_by_run_group")
                label = base_label + 2 * run_group  # 0/1 for 4,8,12 and 2/3 for 6,10,14
            else:
                raise ValueError(f"Unknown label_scheme: {self.label_scheme}")

            onset = int(ev[0])
            if onset + n_samples > raw.n_times:
                continue
            self._indices.append(TrialIndex(path, onset, label, subject_id, run_id))

    def __len__(self) -> int:
        return len(self._indices)

    def _get_picks(self, raw: mne.io.BaseRaw, edf_path: str) -> np.ndarray:
        cached = self._picks_cache.get(edf_path)
        if cached is not None:
            return cached
        picks = mne.pick_types(raw.info, eeg=True, exclude=[])
        if len(picks) != self.expected_n_channels:
            raise ValueError(
                f"Expected {self.expected_n_channels} EEG channels, got {len(picks)} in {edf_path}"
            )
        self._picks_cache[edf_path] = picks
        return picks

    def _load_numpy_sample(self, idx: int) -> tuple[np.ndarray, int]:
        item = self._indices[idx]
        raw = self.raw_cache.get(item.edf_path)
        picks = self._get_picks(raw, item.edf_path)

        sfreq_orig = raw.info["sfreq"]
        n_samples_orig = int(round(self.epoch_duration * sfreq_orig))
        start = item.onset_sample
        stop = start + n_samples_orig
        data = raw.get_data(picks=picks, start=start, stop=stop)
        if self.resample_to_sfreq and abs(sfreq_orig - self.target_sfreq) > 1e-6:
            data = resample(data, up=self.target_sfreq, down=sfreq_orig, npad="auto", axis=-1)
        n_times_target = int(round(self.epoch_duration * self.target_sfreq))
        if data.shape[1] < n_times_target:
            pad = n_times_target - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad)), mode="edge")
        elif data.shape[1] > n_times_target:
            data = data[:, :n_times_target]

        if self._reorder_idx is not None:
            data = data[self._reorder_idx, :]

        # Normalize per-trial, per-channel: (x - mean) / (std + eps).
        # This makes training robust to per-subject amplitude scale differences.
        data = data - data.mean(axis=1, keepdims=True)
        std = data.std(axis=1, keepdims=True)
        data = data / (std + 1e-6)

        return data.astype(np.float32, copy=False), int(item.label)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._indices[idx]
        data, label = self._load_numpy_sample(idx)
        x = torch.from_numpy(data)
        y = torch.tensor(label, dtype=torch.long)
        return {
            "x": x,
            "y": y,
            "subject_id": item.subject_id,
            "run_id": item.run_id,
        }


def resolve_subject_split(
    all_subjects: list[str],
    *,
    train_subjects: list[str] | None,
    val_subjects: list[str] | None,
    test_subjects: list[str] | None,
    seed: int,
    ratios: tuple[float, float, float],
) -> tuple[list[str], list[str], list[str]]:
    if train_subjects is not None and val_subjects is not None and test_subjects is not None:
        tr, va, te = set(train_subjects), set(val_subjects), set(test_subjects)
        if tr & va or tr & te or va & te:
            raise ValueError("Train, val, and test subject lists must be disjoint")
        return sorted(tr), sorted(va), sorted(te)

    subs = list(all_subjects)
    rng = np.random.default_rng(seed)
    rng.shuffle(subs)
    n = len(subs)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    n_train = max(1, min(n - 2, n_train))
    n_val = max(1, min(n - n_train - 1, n_val))
    train = subs[:n_train]
    val = subs[n_train : n_train + n_val]
    test = subs[n_train + n_val :]
    if not test:
        test = [val.pop()]
    return train, val, test
