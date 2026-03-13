from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import torch


CACHE_VERSION = "token-cache-v1"


def _stable_dumps(obj: Dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def build_cache_fingerprint(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    seed: int,
) -> str:
    cache_cfg = dict(dataset_cfg.get("cache", {}))
    fingerprint_payload = {
        "cache_version": str(cache_cfg.get("version", CACHE_VERSION)),
        "dataset_name": str(dataset_cfg.get("name", "")),
        "dataset_path": str(dataset_cfg.get("path", "")),
        "text_column": str(dataset_cfg.get("text_column", "")),
        "label_column": str(dataset_cfg.get("label_column", "")),
        "csv": dict(dataset_cfg.get("csv", {})),
        "split": dict(dataset_cfg.get("split", {})),
        "tokenizer_name": str(preprocessing_cfg.get("tokenizer_name", "roberta-base")),
        "max_seq_len": int(preprocessing_cfg.get("max_seq_len", 512)),
        "remove_urls": bool(preprocessing_cfg.get("remove_urls", True)),
        "remove_mentions": bool(preprocessing_cfg.get("remove_mentions", True)),
        "seed": int(seed),
    }
    digest = hashlib.sha1(_stable_dumps(fingerprint_payload).encode("utf-8")).hexdigest()
    return digest[:16]


def resolve_cache_root(dataset_cfg: Dict[str, Any]) -> Path:
    cache_cfg = dict(dataset_cfg.get("cache", {}))
    root = str(cache_cfg.get("root", "data/cache")).strip()
    return Path(root)


def resolve_cache_dir(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    seed: int,
) -> Path:
    root = resolve_cache_root(dataset_cfg)
    dataset_name = str(dataset_cfg.get("name", "dataset")).strip().lower()
    fingerprint = build_cache_fingerprint(dataset_cfg, preprocessing_cfg, seed)
    return root / dataset_name / fingerprint


def build_expected_metadata(
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    seed: int,
) -> Dict[str, Any]:
    cache_cfg = dict(dataset_cfg.get("cache", {}))
    return {
        "cache_version": str(cache_cfg.get("version", CACHE_VERSION)),
        "dataset_name": str(dataset_cfg.get("name", "")),
        "dataset_path": str(dataset_cfg.get("path", "")),
        "text_column": str(dataset_cfg.get("text_column", "")),
        "label_column": str(dataset_cfg.get("label_column", "")),
        "csv": dict(dataset_cfg.get("csv", {})),
        "split": dict(dataset_cfg.get("split", {})),
        "tokenizer_name": str(preprocessing_cfg.get("tokenizer_name", "roberta-base")),
        "max_seq_len": int(preprocessing_cfg.get("max_seq_len", 512)),
        "remove_urls": bool(preprocessing_cfg.get("remove_urls", True)),
        "remove_mentions": bool(preprocessing_cfg.get("remove_mentions", True)),
        "seed": int(seed),
    }


def metadata_path(cache_dir: Path) -> Path:
    return cache_dir / "metadata.json"


def split_tensor_path(cache_dir: Path, split_name: str) -> Path:
    return cache_dir / f"{split_name}.pt"


def cache_exists(cache_dir: Path) -> bool:
    return (
        metadata_path(cache_dir).exists()
        and split_tensor_path(cache_dir, "train").exists()
        and split_tensor_path(cache_dir, "val").exists()
        and split_tensor_path(cache_dir, "test").exists()
    )


def save_metadata(cache_dir: Path, metadata: Dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_path(cache_dir).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def load_metadata(cache_dir: Path) -> Dict[str, Any]:
    return json.loads(metadata_path(cache_dir).read_text(encoding="utf-8"))


def validate_cache_metadata(
    cache_dir: Path,
    dataset_cfg: Dict[str, Any],
    preprocessing_cfg: Dict[str, Any],
    seed: int,
) -> None:
    expected = build_expected_metadata(dataset_cfg, preprocessing_cfg, seed)
    actual = load_metadata(cache_dir)

    comparable_expected = dict(expected)
    comparable_actual = {
        key: actual.get(key)
        for key in comparable_expected.keys()
    }

    if comparable_expected != comparable_actual:
        raise ValueError(
            "Cache metadata does not match current configuration.\n"
            f"expected={comparable_expected}\nactual={comparable_actual}"
        )


def save_split(cache_dir: Path, split_name: str, payload: Dict[str, torch.Tensor]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, split_tensor_path(cache_dir, split_name))


def load_split(cache_dir: Path, split_name: str) -> Dict[str, torch.Tensor]:
    payload = torch.load(split_tensor_path(cache_dir, split_name), map_location="cpu")
    return {
        "input_ids": payload["input_ids"].to(dtype=torch.long, device="cpu"),
        "attention_mask": payload["attention_mask"].to(dtype=torch.long, device="cpu"),
        "labels": payload["labels"].to(dtype=torch.long, device="cpu"),
    }


def load_cached_bundle(cache_dir: Path) -> Dict[str, Any]:
    metadata = load_metadata(cache_dir)
    return {
        "train": load_split(cache_dir, "train"),
        "val": load_split(cache_dir, "val"),
        "test": load_split(cache_dir, "test"),
        "num_labels": int(metadata["num_labels"]),
        "vocab_size": int(metadata["vocab_size"]),
        "label_mapping": dict(metadata["label_mapping"]),
        "source": str(cache_dir),
        "raw_source": str(metadata["dataset_path"]),
        "tokenizer_name": str(metadata["tokenizer_name"]),
        "tokenizer_backend": str(metadata.get("tokenizer_backend", metadata["tokenizer_name"])),
        "max_seq_len": int(metadata["max_seq_len"]),
        "cache_dir": str(cache_dir),
        "cache_metadata": metadata,
    }