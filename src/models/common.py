from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


def _build_random_roberta(
    *,
    fallback_vocab_size: int,
    fallback_hidden_size: int,
    fallback_max_seq_len: int,
) -> nn.Module:
    from transformers import AutoModel, RobertaConfig

    cfg = RobertaConfig(
        vocab_size=max(100, int(fallback_vocab_size)),
        max_position_embeddings=int(fallback_max_seq_len) + 2,
        hidden_size=int(fallback_hidden_size),
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=int(fallback_hidden_size) * 4,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
    )
    return AutoModel.from_config(cfg)


def load_roberta_encoder(
    encoder_name: str,
    *,
    local_files_only: bool = False,
    allow_random_init: bool = False,
    fallback_vocab_size: int = 50265,
    fallback_hidden_size: int = 128,
    fallback_max_seq_len: int = 512,
) -> nn.Module:
    if local_files_only and allow_random_init:
        return _build_random_roberta(
            fallback_vocab_size=fallback_vocab_size,
            fallback_hidden_size=fallback_hidden_size,
            fallback_max_seq_len=fallback_max_seq_len,
        )
    try:
        from transformers import AutoModel

        return AutoModel.from_pretrained(encoder_name, local_files_only=local_files_only)
    except Exception as exc:
        if not allow_random_init:
            raise RuntimeError(
                f"Could not load pretrained encoder '{encoder_name}'. "
            ) from exc
        return _build_random_roberta(
            fallback_vocab_size=fallback_vocab_size,
            fallback_hidden_size=fallback_hidden_size,
            fallback_max_seq_len=fallback_max_seq_len,
        )


class RobertaBackbone(nn.Module):

    def __init__(self, cfg: Dict[str, Any]) -> None:
        super().__init__()
        encoder_name = str(cfg.get("encoder_name", "roberta-base"))
        self.encoder = load_roberta_encoder(
            encoder_name,
            local_files_only=bool(cfg.get("local_files_only", False)),
            allow_random_init=bool(cfg.get("allow_random_init", False)),
            fallback_vocab_size=int(cfg.get("vocab_size", 50265)),
            fallback_hidden_size=int(cfg.get("fallback_hidden_size", 128)),
            fallback_max_seq_len=int(cfg.get("max_seq_len", 512)),
        )
        self.hidden_size = int(self.encoder.config.hidden_size)
        self.dropout = nn.Dropout(float(cfg.get("dropout", 0.1)))

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state
