from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def save_confusion_matrix_fig(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    class_names: list[str],
    out_path: str | Path,
    normalize: bool = False,
    title: str | None = None,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        disp = cm.astype(np.float64) / row_sums
        fmt = ".2f"
    else:
        disp = cm
        fmt = "d"
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(disp, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.colorbar(im, ax=ax, fraction=0.046)
    for i in range(disp.shape[0]):
        for j in range(disp.shape[1]):
            ax.text(j, i, format(disp[i, j], fmt), ha="center", va="center", color="black" if disp[i, j] < disp.max() * 0.6 else "white")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title or ("Normalized confusion matrix" if normalize else "Confusion matrix"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_training_curves(
    history: dict[str, list[float]],
    out_path: str | Path,
) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history.get("train_loss", [])) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    if history.get("train_loss"):
        axes[0].plot(epochs, history["train_loss"], label="train")
        axes[0].plot(epochs, history["val_loss"], label="val")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].legend()
        axes[0].set_title("Loss")
    if history.get("train_f1"):
        axes[1].plot(epochs, history["train_f1"], label="train macro-F1")
        axes[1].plot(epochs, history["val_f1"], label="val macro-F1")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Macro-F1")
        axes[1].legend()
        axes[1].set_title("Macro-F1")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_metrics_json(payload: dict[str, Any], out_path: str | Path) -> None:
    import json

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
