from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from src.datasets.eegmmidb_dataset import EEGMMIDBDataset, list_subject_ids as list_eegmmi_subject_ids, resolve_subject_split


@dataclass(frozen=True)
class BuiltDatasets:
    train: Dataset
    val: Dataset
    test: Dataset
    reference_channel_names: list[str]


def _dataset_name(cfg: dict[str, Any]) -> str:
    return str(cfg.get("dataset", cfg.get("dataset_name", "eegmmidb"))).strip().lower()


def _dataset_task(cfg: dict[str, Any]) -> str:
    return str(cfg.get("dataset_task", cfg.get("task", ""))).strip().lower()


def build_datasets(cfg: dict[str, Any]) -> BuiltDatasets:
    """
    Returns train/val/test datasets plus the reference channel names from train.
    Keeps model/tokenizer code untouched by matching expected sample dict keys: x,y,subject_id,run_id.
    """
    name = _dataset_name(cfg)
    task = _dataset_task(cfg)
    data_root = Path(cfg["data_root"]).resolve()

    if name in ("eegmmidb", "eegmmi", "eegmmidb_edf"):
        if task in ("mi_4class", "motor_imagery_4class", "fourclass"):
            cfg.setdefault("sfreq", 160)
            cfg.setdefault("epoch_duration", 4.0)
            cfg.setdefault("num_channels", 64)
            cfg.setdefault("num_classes", 4)
            cfg.setdefault("class_names", ["left_hand", "right_hand", "both_feet", "both_hands"])
            cfg.setdefault("label_scheme", "t1t2_by_run_group")
            cfg.setdefault("runs", [4, 6, 8, 10, 12, 14])
            cfg.setdefault("exclude_runs", [1, 2])

        all_subjects = cfg["subjects"] if cfg.get("subjects") is not None else list_eegmmi_subject_ids(data_root)
        if not all_subjects:
            raise FileNotFoundError(f"No subjects found under {data_root}")

        ratios = tuple(cfg.get("subject_split_ratios", [0.7, 0.15, 0.15]))
        train_s, val_s, test_s = resolve_subject_split(
            all_subjects,
            train_subjects=cfg.get("train_subjects"),
            val_subjects=cfg.get("val_subjects"),
            test_subjects=cfg.get("test_subjects"),
            seed=int(cfg.get("subject_split_seed", cfg["seed"])),
            ratios=ratios,
        )

        common_kwargs = dict(
            data_root=data_root,
            epoch_duration=float(cfg["epoch_duration"]),
            sfreq=float(cfg["sfreq"]),
            exclude_runs=list(cfg.get("exclude_runs", [1, 2])),
            canonical_channels=cfg.get("canonical_channels"),
            verify_channel_order=bool(cfg.get("verify_channel_order", True)),
            resample_to_sfreq=bool(cfg.get("resample", True)),
            raw_cache_size=int(cfg.get("raw_cache_size", 6)),
            expected_n_channels=int(cfg.get("num_channels", 64)),
            label_scheme=str(cfg.get("label_scheme", "t1_t2")),
        )

        ds_train = EEGMMIDBDataset(subject_ids=list(train_s), runs=list(cfg["runs"]), **common_kwargs)
        ds_val = EEGMMIDBDataset(subject_ids=list(val_s), runs=list(cfg["runs"]), **common_kwargs)
        ds_test = EEGMMIDBDataset(subject_ids=list(test_s), runs=list(cfg["runs"]), **common_kwargs)
        return BuiltDatasets(ds_train, ds_val, ds_test, ds_train.reference_channel_names)

    raise ValueError(f"Unknown dataset {name!r}. Supported: eegmmidb.")
