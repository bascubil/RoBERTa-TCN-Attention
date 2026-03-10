from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from .common import RobertaBackbone


class RoBERTaRNNClassifier(nn.Module):

    def __init__(self, cfg: Dict[str, Any], num_labels: int) -> None:
        super().__init__()
        self.backbone = RobertaBackbone(cfg)
        variant = str(cfg.get("variant", "roberta_lstm")).strip().lower().replace("-", "_")
        hidden_units = int(cfg.get("hidden_units", 256))
        num_layers = int(cfg.get("num_layers", 1))
        dropout = float(cfg.get("dropout", 0.1))
        bidirectional = variant == "roberta_bilstm"

        if variant in ("roberta_lstm", "roberta_bilstm"):
            self.rnn = nn.LSTM(
                input_size=self.backbone.hidden_size,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self._is_lstm = True
        elif variant == "roberta_gru":
            self.rnn = nn.GRU(
                input_size=self.backbone.hidden_size,
                hidden_size=hidden_units,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self._is_lstm = False
        else:
            raise ValueError(f"Unsupported RNN variant: {variant}")

        out_dim = hidden_units * 2 if bidirectional else hidden_units
        self.classifier = nn.Linear(out_dim, num_labels)
        self.bidirectional = bidirectional

    def _terminal_hidden(self, state: Any) -> torch.Tensor:
        if self._is_lstm:
            h_n = state[0]
        else:
            h_n = state
        if self.bidirectional:
            forward_last = h_n[-2]
            backward_last = h_n[-1]
            return torch.cat([forward_last, backward_last], dim=-1)
        return h_n[-1]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        seq = self.backbone.encode(input_ids=input_ids, attention_mask=attention_mask)
        seq = self.backbone.dropout(seq)
        _, state = self.rnn(seq)
        rep = self._terminal_hidden(state)
        return self.classifier(rep)
