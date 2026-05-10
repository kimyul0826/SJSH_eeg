# EEG Transformer — positional embedding experiments

PyTorch implementation for EEG Motor Imagery classification with tokenizer + Transformer encoder and configurable positional embeddings (`patch_token`, spatial PE, spatiotemporal PE).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

Place PhysioNet EEG MMIDB-style EDF data under the path set in your YAML (`data_root`, default `../files` relative to `configs/`). Subjects should appear as folders `S001`, `S002`, …

## Train

From the `project` directory (so `src` imports resolve):

```bash
python src/train.py --config configs/default.yaml
python src/train.py --config configs/runsets/binary_left_right_runs_4812/cnn_1d.yaml
```

Optional grid:

```bash
python src/run_experiments.py --config configs/default.yaml --output_csv results/grid_results.csv
```

Checkpoints and logs are written under `results/<experiment_slug>/` (see `apply_experiment_paths` in `src/train.py`). Training saves `checkpoints/best.pt` when validation macro-F1 improves.

## Config

YAML keys include `tokenizer_type`, `pe_type`, `patch_size`, `d_model`, dataset runs, subject split, etc. Runsets live under `configs/runsets/`.

## License

Add your license if distributing publicly.
