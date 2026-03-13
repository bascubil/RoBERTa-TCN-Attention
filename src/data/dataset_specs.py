from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import pandas as pd


def _normalize_imdb_labels(values: Sequence[Any]) -> Tuple[List[int], Dict[str, int]]:
    mapping = {"negative": 0, "positive": 1}
    encoded: List[int] = []
    for value in values:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in mapping:
                encoded.append(mapping[key])
                continue
        encoded.append(int(value))
    return encoded, mapping


def _normalize_twitter_labels(values: Sequence[Any]) -> Tuple[List[int], Dict[str, int]]:
    mapping = {
        "negative": 0,
        "neutral": 1,
        "positive": 2,
        "neg": 0,
        "neu": 1,
        "pos": 2,
    }
    encoded: List[int] = []
    for value in values:
        if isinstance(value, str):
            key = value.strip().lower()
            if key in mapping:
                encoded.append(mapping[key])
                continue
        encoded.append(int(value))
    return encoded, {"negative": 0, "neutral": 1, "positive": 2}


def _normalize_sentiment140_labels(values: Sequence[Any]) -> Tuple[List[int], Dict[str, int]]:
    encoded: List[int] = []
    for value in values:
        iv = int(value)
        encoded.append(0 if iv <= 0 else 1)
    return encoded, {"negative": 0, "positive": 1}


def _apply_sentiment140_default_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns and df.shape[1] >= 6:
        df = df.copy()
        df.columns = ["target", "id", "date", "query", "user", "text"] + [
            f"extra_{i}" for i in range(df.shape[1] - 6)
        ]
    return df


def get_dataset_name(dataset_cfg: Dict[str, Any]) -> str:
    return str(dataset_cfg.get("name", "")).strip().lower()


def get_dataset_path(dataset_cfg: Dict[str, Any]) -> Path:
    raw_path = str(dataset_cfg.get("path", "")).strip()
    if not raw_path:
        raise ValueError("dataset.path is empty.")
    path = Path(raw_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")
    return path


def get_dataset_label_mapping(dataset_name: str) -> Dict[str, int]:
    if dataset_name == "imdb":
        return {"negative": 0, "positive": 1}
    if dataset_name == "twitter_us_airline":
        return {"negative": 0, "neutral": 1, "positive": 2}
    if dataset_name == "sentiment140":
        return {"negative": 0, "positive": 1}
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def _normalize_labels_for_dataset(
    dataset_name: str, values: Sequence[Any]
) -> Tuple[List[int], Dict[str, int]]:
    if dataset_name == "imdb":
        return _normalize_imdb_labels(values)
    if dataset_name == "twitter_us_airline":
        return _normalize_twitter_labels(values)
    if dataset_name == "sentiment140":
        return _normalize_sentiment140_labels(values)
    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def iter_standardized_chunks(
    dataset_cfg: Dict[str, Any],
) -> Iterator[Tuple[List[str], List[int], Dict[str, int]]]:
    dataset_name = get_dataset_name(dataset_cfg)
    path = get_dataset_path(dataset_cfg)

    csv_kwargs = dict(dataset_cfg.get("csv", {}))
    chunk_size = int(dataset_cfg.get("cache", {}).get("chunk_size", 50_000))
    if chunk_size <= 0:
        chunk_size = 50_000

    csv_kwargs = dict(csv_kwargs)
    csv_kwargs["chunksize"] = chunk_size

    text_col = str(dataset_cfg.get("text_column", "text"))
    label_col = str(dataset_cfg.get("label_column", "label"))

    chunk_iter = pd.read_csv(path, **csv_kwargs)

    for chunk in chunk_iter:
        if dataset_name == "sentiment140":
            chunk = _apply_sentiment140_default_columns(chunk)

        if text_col not in chunk.columns or label_col not in chunk.columns:
            raise ValueError(
                f"{dataset_name}: required columns not found. "
                f"text='{text_col}', label='{label_col}', columns={list(chunk.columns)}"
            )

        texts = [str(v) for v in chunk[text_col].fillna("").tolist()]
        labels, label_mapping = _normalize_labels_for_dataset(
            dataset_name, chunk[label_col].tolist()
        )
        yield texts, labels, label_mapping


def count_standardized_rows(dataset_cfg: Dict[str, Any]) -> int:
    total = 0
    for texts, _, _ in iter_standardized_chunks(dataset_cfg):
        total += len(texts)
    return total