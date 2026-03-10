"""Model factory for CLI entrypoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .roberta_base import RoBERTaBaseClassifier
from .roberta_rnn import RoBERTaRNNClassifier
from .roberta_tcn import RoBERTaTCNClassifier
from .roberta_tcn_attn import RoBERTaTCNAttentionClassifier

SUPPORTED_VARIANTS = (
    "roberta_base",
    "roberta_lstm",
    "roberta_gru",
    "roberta_bilstm",
    "roberta_tcn",
    "roberta_tcn_attention",
    "roberta_tcn_attn",
)


def normalize_variant(value: str) -> str:
    return value.strip().lower().replace("-", "_")


def create_model(model_cfg: Dict[str, Any], num_labels: int):
    cfg = deepcopy(model_cfg)
    variant = normalize_variant(str(cfg.get("variant", "roberta_tcn_attn")))
    cfg["variant"] = variant
    if variant == "roberta_base":
        return RoBERTaBaseClassifier(cfg=cfg, num_labels=num_labels)
    if variant in ("roberta_lstm", "roberta_gru", "roberta_bilstm"):
        return RoBERTaRNNClassifier(cfg=cfg, num_labels=num_labels)
    if variant == "roberta_tcn":
        return RoBERTaTCNClassifier(cfg=cfg, num_labels=num_labels)
    if variant in ("roberta_tcn_attention", "roberta_tcn_attn"):
        return RoBERTaTCNAttentionClassifier(cfg=cfg, num_labels=num_labels)
    raise ValueError(f"Unsupported model variant '{variant}'. Supported: {SUPPORTED_VARIANTS}")
