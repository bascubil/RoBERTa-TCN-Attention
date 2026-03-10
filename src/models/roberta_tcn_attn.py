"""TCN + attention pooling head with residual fusion.

Per manuscript, the attention-pooled sentence vector is optionally fused with
the *TCN output* at the last valid (non-padding) token position.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.utils.masking import gather_last_valid, masked_softmax

from .common import RobertaBackbone
from .roberta_tcn import TemporalBlock


class RoBERTaTCNAttentionClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_labels: int) -> None:
        super().__init__()
        self.backbone = RobertaBackbone(cfg)
        hidden_units = int(cfg.get("hidden_units", 256))
        kernel_size = int(cfg.get("kernel_size", 8))
        num_layers = int(cfg.get("num_layers", 8))
        dropout = float(cfg.get("dropout", 0.1))
        self.use_residual_fusion = bool(cfg.get("use_residual_fusion", True))

        self.input_proj = nn.Conv1d(self.backbone.hidden_size, hidden_units, kernel_size=1)
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
        self.attn_score = nn.Linear(hidden_units, 1)
        # Residual fusion is an element-wise sum in hidden_units space.
        # Keep an explicit module for clarity/extensibility.
        self.residual_proj = nn.Identity()
        self.post_norm = nn.LayerNorm(hidden_units)
        self.post_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_units, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq = self.backbone.encode(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.backbone.dropout(seq)

        x = self.input_proj(seq.transpose(1, 2))
        x = self.tcn(x).transpose(1, 2)  # [B, L, H]

        attn_logits = self.attn_score(x).squeeze(-1)  # [B, L]
        attn_weights = masked_softmax(attn_logits, attention_mask, dim=-1).unsqueeze(-1)  # [B, L, 1]
        pooled = (x * attn_weights).sum(dim=1)

        if self.use_residual_fusion:
            # Manuscript: fuse with TCN output at the last valid token position.
            last_valid = gather_last_valid(x, attention_mask)
            pooled = pooled + self.residual_proj(last_valid)

        pooled = self.post_dropout(self.post_norm(pooled))
        return self.classifier(pooled)

