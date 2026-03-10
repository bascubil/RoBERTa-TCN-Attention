"""Masking helpers used across model variants."""

from __future__ import annotations

import torch


def last_valid_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return last non-pad position per sample as [B]."""
    if attention_mask.ndim != 2:
        raise ValueError(f"attention_mask must be [B, L], got shape={tuple(attention_mask.shape)}")
    return attention_mask.long().sum(dim=1).clamp_min(1) - 1


def masked_softmax(logits: torch.Tensor, attention_mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Softmax that excludes pad positions and returns 0 for fully-masked rows."""
    if logits.shape != attention_mask.shape:
        raise ValueError(f"logits and attention_mask must have same shape, got {logits.shape} vs {attention_mask.shape}")
    mask = attention_mask.bool()
    masked = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    probs = torch.softmax(masked, dim=dim)
    probs = probs * mask.to(probs.dtype)
    denom = probs.sum(dim=dim, keepdim=True).clamp_min(torch.finfo(probs.dtype).eps)
    return probs / denom


def masked_mean_pool(sequence: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(sequence.dtype)
    summed = (sequence * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom


def gather_last_valid(sequence: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    last_index = last_valid_index(attention_mask)
    batch_idx = torch.arange(sequence.size(0), device=sequence.device)
    return sequence[batch_idx, last_index]
