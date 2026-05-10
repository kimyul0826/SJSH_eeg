from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch
import yaml
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.factory import build_datasets
from src.models.transformer_classifier import build_model, run_dummy_forward_checks, validate_tokenizer_pe_combo
from src.utils.coordinates import build_or_load_coordinates
from src.utils.ddp import is_main_process
from src.utils.metrics import accuracy_macro_f1, count_parameters
from src.utils.plots import save_confusion_matrix_fig, save_metrics_json, save_training_curves
from src.utils.run_log import tee_stdout_txt
from src.utils.seed import set_seed
from src.utils.timing import batch_average_inference_seconds


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config root must be a mapping")
    return cfg


def merge_dict(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    out.update(overrides)
    return out


def normalize_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(v) for v in value]
    raise TypeError(f"Expected str or list, got {type(value)}")


def list_valid_combos(cfg: dict[str, Any]) -> list[tuple[str, str]]:
    tokenizers = normalize_str_list(cfg.get("tokenizer_type"))
    pes = normalize_str_list(cfg.get("pe_type"))
    if not tokenizers or not pes:
        raise ValueError("tokenizer_type and pe_type must be set (string or non-empty list)")
    out: list[tuple[str, str]] = []
    skipped: list[tuple[str, str]] = []
    for tok in tokenizers:
        for pe in pes:
            try:
                validate_tokenizer_pe_combo(tok, pe)
                out.append((tok, pe))
            except ValueError:
                skipped.append((tok, pe))
    if is_main_process() and skipped:
        print(f"Skipping invalid tokenizer/pe combos: {skipped}")
    return out


def apply_experiment_paths(cfg: dict[str, Any], tokenizer_type: str, pe_type: str) -> None:
    root = Path(cfg.get("results_root", "results"))
    slug = f"{tokenizer_type}_{pe_type}"
    if tokenizer_type in ("patch_token", "patch_pooling"):
        slug = f"{slug}_ps{int(cfg.get('patch_size', 0))}"
    exp_dir = (root / slug).resolve()
    cfg["experiment_dir"] = str(exp_dir)
    cfg["checkpoint_dir"] = str(exp_dir / "checkpoints")
    cfg["results_csv"] = str(exp_dir / "experiment_results.csv")
    cfg["plots_dir"] = str(exp_dir / "plots")
    if cfg.get("log_txt") is not False:
        cfg["log_txt"] = str(exp_dir / "training_log.txt")


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    xs = torch.stack([b["x"] for b in batch], dim=0)
    ys = torch.stack([b["y"] for b in batch], dim=0)
    return {
        "x": xs,
        "y": ys,
        "subject_id": [b["subject_id"] for b in batch],
        "run_id": [b["run_id"] for b in batch],
    }


def _device_from_cfg(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unwrap_model(m: torch.nn.Module) -> torch.nn.Module:
    return m


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    desc: str,
    show_progress: bool,
) -> tuple[float, float, float]:
    train = optimizer is not None
    model.train(train)
    non_blocking = device.type == "cuda"
    loss_chunks: list[torch.Tensor] = []
    pred_chunks: list[torch.Tensor] = []
    true_chunks: list[torch.Tensor] = []
    it = loader
    if show_progress:
        nw = int(getattr(loader, "num_workers", 0))
        where = "main thread" if nw == 0 else f"{nw} DataLoader worker(s)"
        print(
            f"{desc}: starting {len(loader)} batches — tqdm updates after the first batch "
            f"(num_workers={nw} → EDF read/resample happens in {where}; first batch can take a while)...",
            flush=True,
        )
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.3)
    for batch in it:
        x = batch["x"].to(device, non_blocking=non_blocking)
        y = batch["y"].to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_chunks.append(loss.detach())
        pred_chunks.append(logits.argmax(dim=-1).detach())
        true_chunks.append(y.detach())
    losses_arr = torch.stack(loss_chunks).cpu().numpy().astype(np.float64)
    y_pred = torch.cat(pred_chunks).cpu().numpy()
    y_true = torch.cat(true_chunks).cpu().numpy()
    acc, f1 = accuracy_macro_f1(y_true, y_pred, num_classes)
    return float(losses_arr.mean()), acc, f1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    desc: str = "eval",
    show_progress: bool = False,
) -> tuple[float, float, float]:
    model.eval()
    non_blocking = device.type == "cuda"
    loss_chunks: list[torch.Tensor] = []
    pred_chunks: list[torch.Tensor] = []
    true_chunks: list[torch.Tensor] = []
    it = loader
    if show_progress:
        print(
            f"{desc}: starting {len(loader)} batches (first batch may be slow)...",
            flush=True,
        )
        it = tqdm(loader, desc=desc, leave=False, dynamic_ncols=True, mininterval=0.3)
    for batch in it:
        x = batch["x"].to(device, non_blocking=non_blocking)
        y = batch["y"].to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)
        loss_chunks.append(loss.detach())
        pred_chunks.append(logits.argmax(dim=-1))
        true_chunks.append(y)
    losses_arr = torch.stack(loss_chunks).cpu().numpy().astype(np.float64)
    y_pred = torch.cat(pred_chunks).cpu().numpy()
    y_true = torch.cat(true_chunks).cpu().numpy()
    acc, f1 = accuracy_macro_f1(y_true, y_pred, num_classes)
    return float(losses_arr.mean()), acc, f1


@torch.no_grad()
def evaluate_with_preds(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
    num_classes: int,
    *,
    show_progress: bool = False,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    non_blocking = device.type == "cuda"
    loss_chunks: list[torch.Tensor] = []
    pred_chunks: list[torch.Tensor] = []
    true_chunks: list[torch.Tensor] = []
    it = loader
    if show_progress:
        print(f"test: starting {len(loader)} batches (first batch may be slow)...", flush=True)
        it = tqdm(loader, desc="test", leave=False, dynamic_ncols=True, mininterval=0.3)
    for batch in it:
        x = batch["x"].to(device, non_blocking=non_blocking)
        y = batch["y"].to(device, non_blocking=non_blocking)
        logits = model(x)
        loss = criterion(logits, y)
        loss_chunks.append(loss.detach())
        pred_chunks.append(logits.argmax(dim=-1))
        true_chunks.append(y)
    losses_arr = torch.stack(loss_chunks).cpu().numpy().astype(np.float64)
    y_pred = torch.cat(pred_chunks).cpu().numpy()
    y_true = torch.cat(true_chunks).cpu().numpy()
    acc, f1 = accuracy_macro_f1(y_true, y_pred, num_classes)
    return float(losses_arr.mean()), acc, f1, y_true, y_pred


def run_training(
    cfg: dict[str, Any],
    *,
    device: torch.device,
) -> dict[str, Any] | None:
    set_seed(int(cfg["seed"]))

    t0 = time.perf_counter()
    if is_main_process():
        ds_name = str(cfg.get("dataset", cfg.get("dataset_name", "eegmmidb")))
        print(f"Setup: building datasets (dataset={ds_name})", flush=True)
        print("  [setup] building train/val/test datasets...", flush=True)
    built = build_datasets(cfg)
    ds_train, ds_val, ds_test = built.train, built.val, built.test
    if is_main_process():
        print(f"  [setup] train dataset: {len(ds_train)} trials in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    if is_main_process():
        print(f"  [setup] val dataset: {len(ds_val)} trials in {time.perf_counter() - t0:.1f}s", flush=True)

    t0 = time.perf_counter()
    if is_main_process():
        print(f"  [setup] test dataset: {len(ds_test)} trials in {time.perf_counter() - t0:.1f}s", flush=True)

    cache_path = cfg.get("coord_cache_path")
    if cache_path:
        cache_path = str(Path(cache_path))
    t0 = time.perf_counter()
    if is_main_process():
        print("  [setup] channel coordinates (cache or compute)...", flush=True)
    coord_loc_path = cfg.get("coord_loc_path")

    coord_meta = build_or_load_coordinates(
        built.reference_channel_names,
        cfg.get("canonical_channels"),
        cache_path,
        coord_loc_path=coord_loc_path,
    )
    if is_main_process():
        print(f"  [setup] coordinates done in {time.perf_counter() - t0:.1f}s", flush=True)
    coord_tensors = {
        "xy": torch.from_numpy(coord_meta["xy_norm"]),
        "xyz": torch.from_numpy(coord_meta["xyz_norm"]),
    }

    nw = int(cfg["num_workers"])
    dl_common: dict[str, Any] = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": nw,
        "collate_fn": collate_batch,
        "pin_memory": device.type == "cuda",
    }
    if nw > 0:
        dl_common["persistent_workers"] = True
        dl_common["prefetch_factor"] = max(2, int(cfg.get("prefetch_factor", 2)))

    train_loader = DataLoader(
        ds_train,
        shuffle=True,
        drop_last=False,
        **dl_common,
    )
    val_loader = DataLoader(ds_val, shuffle=False, **dl_common)
    test_loader = DataLoader(ds_test, shuffle=False, **dl_common)

    t0 = time.perf_counter()
    if is_main_process():
        print("  [setup] building model and moving to device...", flush=True)
    model = build_model(cfg, {"xy": coord_tensors["xy"], "xyz": coord_tensors["xyz"]}).to(device)
    if is_main_process():
        print(
            f"  [setup] model on {device} in {time.perf_counter() - t0:.1f}s "
            f"({count_parameters(model)} parameters)",
            flush=True,
        )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        unwrap_model(model).parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )
    if is_main_process():
        print("  [setup] optimizer ready", flush=True)

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best.pt"

    patience = int(cfg.get("early_stopping_patience", 0))
    min_delta = float(cfg.get("early_stopping_min_delta", 0.0))
    best_f1 = -1.0
    epochs_no_improve = 0
    t_train0 = time.perf_counter()
    num_classes = int(cfg["num_classes"])
    show_pbar = bool(cfg.get("show_progress", True)) and is_main_process()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_f1": [],
        "val_f1": [],
    }

    max_epochs = int(cfg["epochs"])
    if is_main_process():
        print(
            f"Starting training: {max_epochs} epoch(s). First batch may pause briefly (CUDA kernels / disk).",
            flush=True,
        )
    for epoch in range(max_epochs):
        tr_loss, tr_acc, tr_f1 = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            num_classes,
            desc=f"train {epoch + 1}/{max_epochs}",
            show_progress=show_pbar,
        )

        va_loss, va_acc, va_f1 = evaluate(
            model,
            val_loader,
            criterion,
            device,
            num_classes,
            show_progress=show_pbar,
        )

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_f1"].append(tr_f1)
        history["val_f1"].append(va_f1)

        if is_main_process():
            print(
                f"epoch {epoch + 1}/{max_epochs} "
                f"train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
                f"val loss={va_loss:.4f} acc={va_acc:.4f} f1={va_f1:.4f}"
            )

        improved = va_f1 > best_f1 + min_delta
        if improved:
            best_f1 = va_f1
            epochs_no_improve = 0
            if is_main_process():
                torch.save({"model": unwrap_model(model).state_dict(), "cfg": cfg}, best_path)
        else:
            epochs_no_improve += 1

        if patience > 0 and epochs_no_improve >= patience:
            if is_main_process():
                print(f"early stopping at epoch {epoch + 1} (patience={patience})")
            break

    train_time = time.perf_counter() - t_train0

    if best_path.is_file():
        state = torch.load(best_path, map_location=device, weights_only=False)
        unwrap_model(model).load_state_dict(state["model"])

    row_out: dict[str, Any] | None = None
    if is_main_process():
        plots_dir = Path(cfg["plots_dir"])
        plots_dir.mkdir(parents=True, exist_ok=True)
        save_training_curves(history, plots_dir / "training_curves.png")

        te_loss, te_acc, te_f1, y_true, y_pred = evaluate_with_preds(
            model,
            test_loader,
            criterion,
            device,
            num_classes,
            show_progress=show_pbar,
        )

        class_names = cfg.get("class_names")
        if not class_names:
            class_names = [str(i) for i in range(num_classes)]

        save_confusion_matrix_fig(
            y_true,
            y_pred,
            class_names=list(class_names),
            out_path=plots_dir / "confusion_matrix.png",
            normalize=False,
        )
        save_confusion_matrix_fig(
            y_true,
            y_pred,
            class_names=list(class_names),
            out_path=plots_dir / "confusion_matrix_normalized.png",
            normalize=True,
        )

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(num_classes)),
            target_names=list(class_names),
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        save_metrics_json(
            {
                "test_loss": te_loss,
                "accuracy": te_acc,
                "macro_f1": te_f1,
                "classification_report": report,
                "confusion_matrix": cm.tolist(),
            },
            plots_dir / "test_metrics.json",
        )

        batch = next(iter(test_loader))
        xb = batch["x"].to(device)

        def _infer_step() -> None:
            model(xb)

        infer_sec = batch_average_inference_seconds(_infer_step, device)

        params = count_parameters(unwrap_model(model))
        row_out = {
            "tokenizer_type": cfg["tokenizer_type"],
            "pe_type": cfg["pe_type"],
            "patch_size": int(cfg["patch_size"]),
            "d_model": int(cfg["d_model"]),
            "num_layers": int(cfg["num_layers"]),
            "accuracy": te_acc,
            "macro_f1": te_f1,
            "params": params,
            "train_time": train_time,
            "inference_time": infer_sec,
        }
        print(
            f"test loss={te_loss:.4f} acc={te_acc:.4f} macro_f1={te_f1:.4f} "
            f"params={params} train_time={train_time:.2f}s infer/batch={infer_sec:.5f}s"
        )

        write_results_csv(Path(cfg["results_csv"]), row_out)

        master = cfg.get("master_results_csv")
        if master:
            append_results_csv(master, row_out)

    return row_out


def write_results_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {k: row[k] for k in RESULT_CSV_FIELDS}
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS)
        writer.writeheader()
        writer.writerow(out)


RESULT_CSV_FIELDS = [
    "tokenizer_type",
    "pe_type",
    "patch_size",
    "d_model",
    "num_layers",
    "accuracy",
    "macro_f1",
    "params",
    "train_time",
    "inference_time",
]


def append_results_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file()
    out = {k: row[k] for k in RESULT_CSV_FIELDS}
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(out)


def _stdout_log_context(cfg: dict[str, Any]):
    log_txt = cfg.get("log_txt")
    if not log_txt:
        return nullcontext()
    return tee_stdout_txt(
        log_txt,
        append=bool(cfg.get("log_txt_append", False)),
    )


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        message=".*enable_nested_tensor.*",
        category=UserWarning,
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--dummy", action="store_true", help="Run shape checks and exit")
    parser.add_argument("--override", type=str, default=None, help="JSON object of config overrides")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = root / args.config
    cfg = load_yaml_config(cfg_path)
    if args.override:
        cfg = merge_dict(cfg, json.loads(args.override))

    if args.dummy:
        dev = _device_from_cfg(str(cfg.get("device", "cpu")))
        run_dummy_forward_checks(dev)
        print("dummy forward ok")
        return

    device = _device_from_cfg(str(cfg.get("device", "cuda")))

    combos = list_valid_combos(cfg)
    if not combos:
        raise ValueError("No valid tokenizer_type / pe_type combinations")

    for tok, pe in combos:
        run_cfg = deepcopy(cfg)
        run_cfg["tokenizer_type"] = tok
        run_cfg["pe_type"] = pe
        apply_experiment_paths(run_cfg, tok, pe)
        if is_main_process():
            print(f"=== experiment: {tok} + {pe} -> {run_cfg['experiment_dir']} ===")
        with _stdout_log_context(run_cfg):
            run_training(run_cfg, device=device)


if __name__ == "__main__":
    main()
