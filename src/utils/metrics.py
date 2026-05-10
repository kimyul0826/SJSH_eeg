from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score


def accuracy_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> tuple[float, float]:
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0))
    return acc, macro_f1


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
