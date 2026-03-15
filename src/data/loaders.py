from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

from .cache import (
    build_expected_metadata,
    cache_exists,
    load_cached_bundle,
    resolve_cache_dir,
    save_metadata,
    save_split,
    validate_cache_metadata,
)
from .dataset_specs import (
    count_standardized_rows,
    get_dataset_label_mapping,
    get_dataset_name,
    iter_standardized_chunks,
)
from .preprocessing import clean_text


class _OfflineFallbackTokenizer:

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
            raise ValueError("return_tensors='pt'")

        input_ids: List[List[int]] = []
        masks: List[List[int]] = []

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


def _build_assignment_tensor(
    size: int,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> torch.Tensor:
    train_idx, val_idx, test_idx = _split_indices(size, train_ratio, val_ratio, seed)
    assignment = torch.empty(size, dtype=torch.uint8)
    assignment[train_idx] = 0
    assignment[val_idx] = 1
    assignment[test_idx] = 2
    return assignment


def _empty_split(max_seq_len: int) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": torch.zeros((0, max_seq_len), dtype=torch.long),
        "attention_mask": torch.zeros((0, max_seq_len), dtype=torch.long),
        "labels": torch.zeros((0,), dtype=torch.long),
    }


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
        "input_ids": encoded["input_ids"].to(dtype=torch.long, device="cpu"),
        "attention_mask": encoded["attention_mask"].to(dtype=torch.long, device="cpu"),
    }


def _concat_or_empty(chunks: List[torch.Tensor], shape_tail: Tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    if not chunks:
        return torch.zeros((0, *shape_tail), dtype=dtype)
    return torch.cat(chunks, dim=0)


def _finalize_split(
    split_accumulator: Dict[str, List[torch.Tensor]],
    max_seq_len: int,
) -> Dict[str, torch.Tensor]:
    return {
        "input_ids": _concat_or_empty(split_accumulator["input_ids"], (max_seq_len,), torch.long),
        "attention_mask": _concat_or_empty(split_accumulator["attention_mask"], (max_seq_len,), torch.long),
        "labels": _concat_or_empty(split_accumulator["labels"], (), torch.long),
    }


def _load_tokenizer(tokenizer_name: str, sample_mode: bool) -> Tuple[Any, str]:
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            use_fast=True,
            local_files_only=sample_mode,
        )
        return tokenizer, tokenizer_name
    except Exception as exc:
        if sample_mode:
            return _OfflineFallbackTokenizer(), "offline-fallback"
        raise RuntimeError(
            f"Failed to load tokenizer '{tokenizer_name}'. "
        ) from exc


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

    texts: List[str] = []
    labels: List[int] = []
    for i in range(max(8, sample_size)):
        label = classes[i % len(classes)]
        text = base_texts[label % len(base_texts)] + f" sample_{i}"
        texts.append(text)
        labels.append(label)
    return texts, labels, mapping


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


def _build_sample_bundle(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    tokenizer_name = str(preprocessing_cfg.get("tokenizer_name", "roberta-base"))
    max_seq_len = int(preprocessing_cfg.get("max_seq_len", 512))
    remove_urls = bool(preprocessing_cfg.get("remove_urls", True))
    remove_mentions = bool(preprocessing_cfg.get("remove_mentions", True))

    dataset_name = get_dataset_name(dataset_cfg)
    sample_size = int(dataset_cfg.get("sample_size", 64))

    texts, labels, label_mapping = _build_sample_records(dataset_name=dataset_name, sample_size=sample_size)
    texts = [clean_text(t, remove_urls=remove_urls, remove_mentions=remove_mentions) for t in texts]

    split_cfg = dataset_cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.9))
    val_ratio = float(split_cfg.get("val", 0.05))
    train_idx, val_idx, test_idx = _split_indices(len(texts), train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    tokenizer, tokenizer_backend = _load_tokenizer(tokenizer_name=tokenizer_name, sample_mode=True)

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
        "source": "sample-mode",
        "raw_source": "sample-mode",
        "tokenizer_name": tokenizer_name,
        "tokenizer_backend": tokenizer_backend,
        "max_seq_len": max_seq_len,
        "cache_dir": None,
        "cache_metadata": None,
    }


def prepare_dataset_cache(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    seed: int,
    force_rebuild: bool = False,
) -> Path:
    del training_cfg

    sample_mode = bool(dataset_cfg.get("sample_mode", False) or dataset_cfg.get("use_synthetic_data", False))
    if sample_mode:
        raise ValueError("does not support sample_mode=true.")

    tokenizer_name = str(preprocessing_cfg.get("tokenizer_name", "roberta-base"))
    max_seq_len = int(preprocessing_cfg.get("max_seq_len", 512))
    remove_urls = bool(preprocessing_cfg.get("remove_urls", True))
    remove_mentions = bool(preprocessing_cfg.get("remove_mentions", True))
    cache_cfg = dict(dataset_cfg.get("cache", {}))
    tokenize_batch_size = int(cache_cfg.get("tokenize_batch_size", 1024))
    if tokenize_batch_size <= 0:
        tokenize_batch_size = 1024

    split_cfg = dataset_cfg.get("split", {})
    train_ratio = float(split_cfg.get("train", 0.9))
    val_ratio = float(split_cfg.get("val", 0.05))

    cache_dir = resolve_cache_dir(dataset_cfg, preprocessing_cfg, seed)

    if cache_exists(cache_dir) and not force_rebuild:
        validate_cache_metadata(cache_dir, dataset_cfg, preprocessing_cfg, seed)
        return cache_dir

    dataset_name = get_dataset_name(dataset_cfg)
    total_rows = count_standardized_rows(dataset_cfg)
    if total_rows <= 0:
        raise ValueError(f"No valid rows found for dataset '{dataset_name}'.")

    assignment = _build_assignment_tensor(
        size=total_rows,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        seed=seed,
    )

    tokenizer, tokenizer_backend = _load_tokenizer(tokenizer_name=tokenizer_name, sample_mode=False)
    vocab_size = int(getattr(tokenizer, "vocab_size", 50265))
    label_mapping = get_dataset_label_mapping(dataset_name)
    num_labels = len(set(label_mapping.values()))

    accumulators: Dict[str, Dict[str, List[torch.Tensor]]] = {
        "train": {"input_ids": [], "attention_mask": [], "labels": []},
        "val": {"input_ids": [], "attention_mask": [], "labels": []},
        "test": {"input_ids": [], "attention_mask": [], "labels": []},
    }
    split_counts = {"train": 0, "val": 0, "test": 0}

    global_row_index = 0

    for raw_texts, raw_labels, chunk_label_mapping in iter_standardized_chunks(dataset_cfg):
        if chunk_label_mapping:
            label_mapping = {str(k): int(v) for k, v in chunk_label_mapping.items()}
            num_labels = len(set(label_mapping.values()))

        cleaned_texts = [
            clean_text(text, remove_urls=remove_urls, remove_mentions=remove_mentions)
            for text in raw_texts
        ]

        per_split_texts = {"train": [], "val": [], "test": []}
        per_split_labels = {"train": [], "val": [], "test": []}

        for offset, (text, label) in enumerate(zip(cleaned_texts, raw_labels)):
            split_id = int(assignment[global_row_index + offset].item())
            if split_id == 0:
                split_name = "train"
            elif split_id == 1:
                split_name = "val"
            else:
                split_name = "test"

            per_split_texts[split_name].append(text)
            per_split_labels[split_name].append(int(label))
            split_counts[split_name] += 1

        global_row_index += len(cleaned_texts)

        for split_name in ("train", "val", "test"):
            split_texts = per_split_texts[split_name]
            split_labels = per_split_labels[split_name]
            if not split_texts:
                continue

            for start in range(0, len(split_texts), tokenize_batch_size):
                end = min(start + tokenize_batch_size, len(split_texts))
                batch_texts = split_texts[start:end]
                batch_labels = split_labels[start:end]

                encoded = _tokenize_texts(tokenizer, batch_texts, max_seq_len=max_seq_len)
                accumulators[split_name]["input_ids"].append(encoded["input_ids"])
                accumulators[split_name]["attention_mask"].append(encoded["attention_mask"])
                accumulators[split_name]["labels"].append(torch.tensor(batch_labels, dtype=torch.long))

    if global_row_index != total_rows:
        raise RuntimeError(
            f"Row counting mismatch while building cache: expected {total_rows}, got {global_row_index}"
        )

    train_payload = _finalize_split(accumulators["train"], max_seq_len=max_seq_len)
    val_payload = _finalize_split(accumulators["val"], max_seq_len=max_seq_len)
    test_payload = _finalize_split(accumulators["test"], max_seq_len=max_seq_len)

    metadata = build_expected_metadata(dataset_cfg, preprocessing_cfg, seed)
    metadata.update(
        {
            "tokenizer_backend": tokenizer_backend,
            "vocab_size": vocab_size,
            "num_labels": num_labels,
            "label_mapping": {str(k): int(v) for k, v in label_mapping.items()},
            "row_counts": {
                "total": int(total_rows),
                "train": int(split_counts["train"]),
                "val": int(split_counts["val"]),
                "test": int(split_counts["test"]),
            },
        }
    )

    save_split(cache_dir, "train", train_payload)
    save_split(cache_dir, "val", val_payload)
    save_split(cache_dir, "test", test_payload)
    save_metadata(cache_dir, metadata)

    return cache_dir


def build_data_bundle(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    sample_mode = bool(dataset_cfg.get("sample_mode", False) or dataset_cfg.get("use_synthetic_data", False))
    if sample_mode:
        return _build_sample_bundle(dataset_cfg, preprocessing_cfg, seed=seed)

    cache_cfg = dict(dataset_cfg.get("cache", {}))
    cache_enabled = bool(cache_cfg.get("enabled", True))
    build_if_missing = bool(cache_cfg.get("build_if_missing", False))

    if not cache_enabled:
        raise ValueError("Set dataset.cache.enabled=true")

    cache_dir = resolve_cache_dir(dataset_cfg, preprocessing_cfg, seed)

    if cache_exists(cache_dir):
        validate_cache_metadata(cache_dir, dataset_cfg, preprocessing_cfg, seed)
        return load_cached_bundle(cache_dir)

    if build_if_missing:
        prepare_dataset_cache(
            dataset_cfg=dataset_cfg,
            preprocessing_cfg=preprocessing_cfg,
            training_cfg=training_cfg,
            seed=seed,
            force_rebuild=False,
        )
        return load_cached_bundle(cache_dir)

    raise FileNotFoundError(
        f"Cache not found at: {cache_dir}\n"
        "Run scripts/prepare_dataset_cache.py"
    )