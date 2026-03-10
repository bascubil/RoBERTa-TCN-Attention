"""TCN head over shared RoBERTa last_hidden_state.

This implementation follows the manuscript description:
  - dilated *causal* 1D convolutions (no future token leakage)
  - increasing dilation factors (exponentially expanding receptive field)
  - residual blocks with (Conv → Norm → ReLU → Dropout) × 2
  - optional 1×1 projection when channel dimensions differ

The module preserves sequence length so the upstream attention mask can be
reused without modification.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.masking import masked_mean_pool

from .common import RobertaBackbone


class TemporalBlock(nn.Module):
    """A single TCN residual block.

    Notes
    -----
    - Inputs/outputs are shaped as [B, C, L].
    - Causal padding is applied on the *left* only.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size < 1:
            raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
        if dilation < 1:
            raise ValueError(f"dilation must be >= 1, got {dilation}")

        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)
        self.left_pad = (self.kernel_size - 1) * self.dilation

        # Causal dilated convolutions (padding is applied manually in forward).
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
        )

        # Activation normalization (per time-step over channels).
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(float(dropout))

        # Residual projection (1×1 conv) if channels differ.
        self.downsample: nn.Module
        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.downsample = nn.Identity()

    def _causal_dilated_conv(self, conv: nn.Conv1d, x: torch.Tensor) -> torch.Tensor:
        # Left pad only so output at time t depends on <= t.
        if self.left_pad > 0:
            x = F.pad(x, (self.left_pad, 0))
        return conv(x)

    @staticmethod
    def _layer_norm_channels(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
        # x: [B, C, L] -> [B, L, C] for LayerNorm
        return norm(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._causal_dilated_conv(self.conv1, x)
        y = self._layer_norm_channels(y, self.norm1)
        y = self.relu(y)
        y = self.dropout(y)

        y = self._causal_dilated_conv(self.conv2, y)
        y = self._layer_norm_channels(y, self.norm2)
        y = self.relu(y)
        y = self.dropout(y)

        return self.downsample(x) + y


class RoBERTaTCNClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_labels: int) -> None:
        super().__init__()
        self.backbone = RobertaBackbone(cfg)
        hidden_units = int(cfg.get("hidden_units", 256))
        kernel_size = int(cfg.get("kernel_size", 8))
        num_layers = int(cfg.get("num_layers", 8))
        dropout = float(cfg.get("dropout", 0.1))

        self.input_proj = nn.Conv1d(self.backbone.hidden_size, hidden_units, kernel_size=1)
        # Increasing dilation factors: 1, 2, 4, ...
        self.tcn = nn.Sequential(
            *[
                TemporalBlock(
                    in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=kernel_size,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(num_layers)
            ]
        )
        self.classifier = nn.Linear(hidden_units, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq = self.backbone.encode(input_ids=input_ids, attention_mask=attention_mask)
        x = self.input_proj(self.backbone.dropout(seq).transpose(1, 2))
        x = self.tcn(x).transpose(1, 2)
        pooled = masked_mean_pool(x, attention_mask)
        return self.classifier(pooled)

