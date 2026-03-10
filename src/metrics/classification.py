from __future__ import annotations

from typing import Iterable, List

import torch


def _to_list(values) -> List[int]:
    if isinstance(values, torch.Tensor):
        return [int(v) for v in values.detach().cpu().tolist()]
    if isinstance(values, Iterable):
        return [int(v) for v in values]
    raise TypeError(f"Unsupported values type: {type(values)}")


def accuracy(preds, labels) -> float:
    p = _to_list(preds)
    y = _to_list(labels)
    if not y:
        return 0.0
    correct = sum(1 for pi, yi in zip(p, y) if pi == yi)
    return correct / len(y)


def macro_f1(preds, labels) -> float:
    p = _to_list(preds)
    y = _to_list(labels)
    if not y:
        return 0.0
    classes = sorted(set(y) | set(p))
    f1_scores = []
    for c in classes:
        tp = sum(1 for pi, yi in zip(p, y) if pi == c and yi == c)
        fp = sum(1 for pi, yi in zip(p, y) if pi == c and yi != c)
        fn = sum(1 for pi, yi in zip(p, y) if pi != c and yi == c)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            f1_scores.append(0.0)
        else:
            f1_scores.append(2.0 * precision * recall / (precision + recall))
    return sum(f1_scores) / len(f1_scores)

