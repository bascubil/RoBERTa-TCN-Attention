from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from src.utils.masking import gather_last_valid

from .common import RobertaBackbone


class RoBERTaBaseClassifier(nn.Module):
    def __init__(self, cfg: Dict[str, Any], num_labels: int) -> None:
        super().__init__()
        self.backbone = RobertaBackbone(cfg)
        hidden_units = int(cfg.get("hidden_units", self.backbone.hidden_size))
        self.head = nn.Sequential(
            nn.Linear(self.backbone.hidden_size, hidden_units),
            nn.ReLU(),
            nn.Dropout(float(cfg.get("dropout", 0.1))),
            nn.Linear(hidden_units, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq = self.backbone.encode(input_ids=input_ids, attention_mask=attention_mask)
        pooled = gather_last_valid(seq, attention_mask)
        return self.head(self.backbone.dropout(pooled))

