from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from .preprocessing import clean_text


class _OfflineFallbackTokenizer:
    """Minimal tokenizer used only for offline sample mode smoke tests."""

    def __init__(self, vocab_size: int = 50265) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 1
        self.bos_token_id = 0
        self.eos_token_id = 2

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
        return 3 + (int(digest[:8], 16) % max(1, self.vocab_size - 3))

    def __call__(
        self,
        texts: Sequence[str],
        truncation: bool,
        padding: str,
        max_length: int,
        return_attention_mask: bool,
        return_tensors: str,
    ) -> Dict[str, torch.Tensor]:
        if return_tensors != "pt":
            raise ValueError("Fallback tokenizer only supports return_tensors='pt'")
        input_ids = []
        masks = []
        for text in texts:
            token_ids = [self.bos_token_id] + [self._token_to_id(tok) for tok in str(text).split()] + [self.eos_token_id]
            if truncation:
                token_ids = token_ids[:max_length]
            mask = [1] * len(token_ids)
            if padding == "max_length" and len(token_ids) < max_length:
                pad_len = max_length - len(token_ids)
                token_ids += [self.pad_token_id] * pad_len
                mask += [0] * pad_len
            input_ids.append(token_ids)
            masks.append(mask)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long) if return_attention_mask else None,
        }


def _split_indices(size: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    indices = list(range(size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_end = int(size * train_ratio)
    val_end = train_end + int(size * val_ratio)
    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]
    return train_idx, val_idx, test_idx


def _tokenize_texts(tokenizer: Any, texts: Sequence[str], max_seq_len: int) -> Dict[str, torch.Tensor]:
    if len(texts) == 0:
        return {
            "input_ids": torch.zeros((0, max_seq_len), dtype=torch.long),
            "attention_mask": torch.zeros((0, max_seq_len), dtype=torch.long),
        }
    encoded = tokenizer(
        list(texts),
        truncation=True,
        padding="max_length",
        max_length=max_seq_len,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return {
        "input_ids": encoded["input_ids"].to(dtype=torch.long),
        "attention_mask": encoded["attention_mask"].to(dtype=torch.long),
    }


def _make_split(
    texts: Sequence[str],
    labels: Sequence[int],
    indices: Sequence[int],
    tokenizer: Any,
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    selected_texts = [texts[i] for i in indices]
    selected_labels = [labels[i] for i in indices]
    tokens = _tokenize_texts(tokenizer, selected_texts, max_seq_len=max_seq_len)
    tokens["labels"] = torch.tensor(selected_labels, dtype=torch.long)
    return tokens


def _load_tokenizer(tokenizer_name: str, sample_mode: bool) -> Tuple[Any, str]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True, local_files_only=sample_mode)
        return tokenizer, tokenizer_name
    except Exception as exc:
        if sample_mode:
            return _OfflineFallbackTokenizer(), "offline-fallback"
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_name}'. "
            "Ensure internet/cache access, or run with sample_mode=true for offline smoke tests."
        ) from exc


def _normalize_imdb_labels(values: Sequence[Any]) -> Tuple[List[int], Dict[str, int]]:
    mapping = {"negative": 0, "positive": 1}
    encoded = []
    for value in values:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in mapping:
                encoded.append(mapping[key])
                continue
        encoded.append(int(value))
    return encoded, mapping


def _normalize_twitter_labels(values: Sequence[Any]) -> Tuple[List[int], Dict[str, int]]:
    mapping = {"negative": 0, "neutral": 1, "positive": 2, "neg": 0, "neu": 1, "pos": 2}
    encoded = []
    for value in values:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in mapping:
                encoded.append(mapping[key])
                continue
        encoded.append(int(value))
    return encoded, {"negative": 0, "neutral": 1, "positive": 2}


def _normalize_sentiment140_labels(values: Sequence[Any]) -> Tuple[List[int], Dict[str, int]]:
    encoded = []
    for value in values:
        iv = int(value)
        encoded.append(0 if iv <= 0 else 1)
    return encoded, {"negative": 0, "positive": 1}


def _load_imdb_records(path: Path, dataset_cfg: Dict[str, Any]) -> Tuple[List[str], List[int], Dict[str, int]]:
    import pandas as pd

    csv_kwargs = dict(dataset_cfg.get("csv", {}))
    df = pd.read_csv(path, **csv_kwargs)
    text_col = str(dataset_cfg.get("text_column", "review"))
    label_col = str(dataset_cfg.get("label_column", "sentiment"))
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"IMDb columns not found: text='{text_col}' label='{label_col}' in columns={list(df.columns)}")
    texts = [str(v) for v in df[text_col].fillna("").tolist()]
    labels, mapping = _normalize_imdb_labels(df[label_col].tolist())
    return texts, labels, mapping


def _load_twitter_records(path: Path, dataset_cfg: Dict[str, Any]) -> Tuple[List[str], List[int], Dict[str, int]]:
    import pandas as pd

    csv_kwargs = dict(dataset_cfg.get("csv", {}))
    df = pd.read_csv(path, **csv_kwargs)
    text_col = str(dataset_cfg.get("text_column", "text"))
    label_col = str(dataset_cfg.get("label_column", "airline_sentiment"))
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Twitter columns not found: text='{text_col}' label='{label_col}' in columns={list(df.columns)}"
        )
    texts = [str(v) for v in df[text_col].fillna("").tolist()]
    labels, mapping = _normalize_twitter_labels(df[label_col].tolist())
    return texts, labels, mapping


def _load_sentiment140_records(path: Path, dataset_cfg: Dict[str, Any]) -> Tuple[List[str], List[int], Dict[str, int]]:
    import pandas as pd

    csv_kwargs = dict(dataset_cfg.get("csv", {}))
    df = pd.read_csv(path, **csv_kwargs)

    if "text" not in df.columns and df.shape[1] >= 6:
        df.columns = ["target", "id", "date", "query", "user", "text"] + [f"extra_{i}" for i in range(df.shape[1] - 6)]

    text_col = str(dataset_cfg.get("text_column", "text"))
    label_col = str(dataset_cfg.get("label_column", "target"))
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Sentiment140 columns not found: text='{text_col}' label='{label_col}' in columns={list(df.columns)}"
        )

    texts = [str(v) for v in df[text_col].fillna("").tolist()]
    labels, mapping = _normalize_sentiment140_labels(df[label_col].tolist())
    return texts, labels, mapping


def _build_sample_records(dataset_name: str, sample_size: int) -> Tuple[List[str], List[int], Dict[str, int]]:
    if dataset_name == "twitter_us_airline":
        classes = [0, 1, 2]
        mapping = {"negative": 0, "neutral": 1, "positive": 2}
        base_texts = [
            "@airline Flight delayed again... http://example.com",
            "The crew was okay, average service.",
            "Loved the flight! Great support from @agent",
        ]
    else:
        classes = [0, 1]
        mapping = {"negative": 0, "positive": 1}
        base_texts = [
            "Terrible experience. Will not return. http://bad.example",
            "Excellent and enjoyable review from @critic.",
        ]

    texts = []
    labels = []
    for i in range(max(8, sample_size)):
        label = classes[i % len(classes)]
        text = base_texts[label % len(base_texts)] + f" sample_{i}"
        texts.append(text)
        labels.append(label)
    return texts, labels, mapping


def _load_records(dataset_cfg: Dict[str, Any]) -> Tuple[List[str], List[int], Dict[str, int], str]:
    dataset_name = str(dataset_cfg.get("name", "")).strip().lower()
    dataset_path = str(dataset_cfg.get("path", "")).strip()
    sample_mode = bool(dataset_cfg.get("sample_mode", False) or dataset_cfg.get("use_synthetic_data", False))
    sample_size = int(dataset_cfg.get("sample_size", 64))

    if sample_mode:
        texts, labels, mapping = _build_sample_records(dataset_name=dataset_name, sample_size=sample_size)
        return texts, labels, mapping, "sample-mode"

    if not dataset_path:
        raise ValueError("Dataset path is empty. Set dataset.path or enable dataset.sample_mode=true.")
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    if dataset_name == "imdb":
        texts, labels, mapping = _load_imdb_records(path, dataset_cfg)
    elif dataset_name == "twitter_us_airline":
        texts, labels, mapping = _load_twitter_records(path, dataset_cfg)
    elif dataset_name == "sentiment140":
        texts, labels, mapping = _load_sentiment140_records(path, dataset_cfg)
    else:
        raise ValueError(f"Unsupported dataset name '{dataset_name}'. Expected imdb/twitter_us_airline/sentiment140.")
    return texts, labels, mapping, str(path)


def build_data_bundle(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    del training_cfg  # reserved for future loader behavior

    tokenizer_name = str(preprocessing_cfg.get("tokenizer_name", "roberta-base"))
    max_seq_len = int(preprocessing_cfg.get("max_seq_len", 512))
    remove_urls = bool(preprocessing_cfg.get("remove_urls", True))
    remove_mentions = bool(preprocessing_cfg.get("remove_mentions", True))

    texts, labels, label_mapping, source = _load_records(dataset_cfg)
    texts = [clean_text(t, remove_urls=remove_urls, remove_mentions=remove_mentions) for t in texts]

    split_cfg = dataset_cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.9))
    val_ratio = float(split_cfg.get("val", 0.05))
    train_idx, val_idx, test_idx = _split_indices(len(texts), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    sample_mode = bool(dataset_cfg.get("sample_mode", False) or dataset_cfg.get("use_synthetic_data", False))
    tokenizer, tokenizer_backend = _load_tokenizer(tokenizer_name=tokenizer_name, sample_mode=sample_mode)

    train = _make_split(texts, labels, train_idx, tokenizer=tokenizer, max_seq_len=max_seq_len)
    val = _make_split(texts, labels, val_idx, tokenizer=tokenizer, max_seq_len=max_seq_len)
    test = _make_split(texts, labels, test_idx, tokenizer=tokenizer, max_seq_len=max_seq_len)

    vocab_size = int(getattr(tokenizer, "vocab_size", 50265))
    num_labels = len(set(labels)) if labels else int(dataset_cfg.get("num_labels", 2))
    return {
        "train": train,
        "val": val,
        "test": test,
        "num_labels": num_labels,
        "vocab_size": vocab_size,
        "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
        "source": source,
        "tokenizer_name": tokenizer_name,
        "tokenizer_backend": tokenizer_backend,
        "max_seq_len": max_seq_len,
    }
