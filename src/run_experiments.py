from __future__ import annotations

import argparse
import sys
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.train import append_results_csv, apply_experiment_paths, load_yaml_config, run_training
from src.utils.ddp import is_main_process
from src.utils.run_log import tee_stdout_txt


def _device_from_yaml(base: dict) -> torch.device:
    s = str(base.get("device", "cuda"))
    if s == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_csv", type=str, default="results/grid_results.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        cfg_path = root / args.config
    base = load_yaml_config(cfg_path)

    device = _device_from_yaml(base)

    spatial_tokenizers = ["linear", "cnn", "frequency", "patch_pooling"]
    spatial_pes = ["none", "1d", "2d", "3d"]
    st_tokenizer = "patch_token"
    patch_sizes = [40, 80]
    st_pes = ["none", "1d_t", "2d_t", "3d_t"]

    rows: list[dict] = []
    for tok in spatial_tokenizers:
        for pe in spatial_pes:
            cfg = deepcopy(base)
            cfg["tokenizer_type"] = tok
            cfg["pe_type"] = pe
            if tok != "patch_pooling":
                cfg["patch_size"] = int(base.get("patch_size", 40))
            apply_experiment_paths(cfg, tok, pe)
            if is_main_process():
                print(f"=== grid: {tok} + {pe} -> {cfg['experiment_dir']} ===")
            ctx = tee_stdout_txt(cfg["log_txt"], append=False) if cfg.get("log_txt") else nullcontext()
            with ctx:
                row = run_training(cfg, device=device)
            if row:
                rows.append(row)
                append_results_csv(args.output_csv, row)

    for ps in patch_sizes:
        for pe in st_pes:
            cfg = deepcopy(base)
            cfg["tokenizer_type"] = st_tokenizer
            cfg["pe_type"] = pe
            cfg["patch_size"] = int(ps)
            apply_experiment_paths(cfg, st_tokenizer, pe)
            if is_main_process():
                print(f"=== grid: {st_tokenizer} + {pe} ps={ps} -> {cfg['experiment_dir']} ===")
            ctx = tee_stdout_txt(cfg["log_txt"], append=False) if cfg.get("log_txt") else nullcontext()
            with ctx:
                row = run_training(cfg, device=device)
            if row:
                rows.append(row)
                append_results_csv(args.output_csv, row)

    if is_main_process():
        print(f"completed {len(rows)} runs; summary appended to {args.output_csv}")


if __name__ == "__main__":
    main()
