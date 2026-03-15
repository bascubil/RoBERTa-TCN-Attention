from __future__ import annotations

import csv
import math
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_data_bundle
from src.metrics import accuracy, macro_f1
from src.models import create_model
from src.utils.config import deep_merge, load_composed_config, resolve_config_path
from src.utils.seed import set_seed
from src.utils.throughput import ThroughputTimer


DATASET_KEYS = ("imdb", "twitter_us_airline", "sentiment140")
DATASET_DISPLAY = {
    "imdb": "IMDb",
    "twitter_us_airline": "Twitter",
    "sentiment140": "Sentiment140",
}

TABLE1_MODELS = [
    ("RoBERTa-base", "roberta_base"),
    ("RoBERTa-LSTM", "roberta_lstm"),
    ("RoBERTa-BiLSTM", "roberta_bilstm"),
    ("RoBERTa-GRU", "roberta_gru"),
    ("RoBERTa-TCN", "roberta_tcn"),
    ("RoBERTa-TCN-Attention", "roberta_tcn_attn"),
]

TABLE2_MODELS = [
    ("TCN", "roberta_tcn"),
    ("TCN-Attn (without residual connection)", "roberta_tcn_attn_no_residual"),
    ("TCN-Attn (with residual connection)", "roberta_tcn_attn_residual"),
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_float(value: float) -> str:
    return f"{float(value):.2f}"


def format_lr(value: float) -> str:
    return f"{float(value):.0e}".replace("e-0", "e-").replace("e+0", "e+")


def canonical_hidden(value: Any) -> str:
    if value in (None, "-", ""):
        return "-"
    return str(int(value))


def default_device(requested: str | None = None) -> str:
    if requested:
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    exp_path = resolve_config_path(config_path, base_dir=REPO_ROOT, repo_root=REPO_ROOT)
    return load_composed_config(exp_path, repo_root=REPO_ROOT)


def load_grid_settings(config_path: str) -> Tuple[List[int], List[float]]:
    cfg = load_experiment_config(config_path)
    training_cfg = cfg.get("training", {})
    grid_cfg = training_cfg.get("grid", {})
    hidden_grid = [int(v) for v in grid_cfg.get("hidden_units", [128, 256, 512])]
    lr_grid = [float(v) for v in grid_cfg.get("learning_rate", [1e-4, 1e-5, 1e-6])]
    return hidden_grid, lr_grid


def compose_runtime_config(
    *,
    config_path: str,
    dataset: str,
    variant: str,
    hidden_units: int | None,
    learning_rate: float | None,
    max_seq_len: int | None,
    epochs: int | None,
    batch_size: int | None,
    sample_mode: bool,
    sample_size: int,
    seed: int,
    device: str | None,
    cache_root: str | None = None,
    build_cache_if_missing: bool | None = None,
    cache_enabled: bool | None = None,
    model_overrides: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    cfg = deep_merge({}, load_experiment_config(config_path))

    dataset_cfg_path = REPO_ROOT / "configs" / "datasets" / f"{dataset}.yaml"
    model_cfg_map = {
        "roberta_base": "roberta_base.yaml",
        "roberta_lstm": "roberta_lstm.yaml",
        "roberta_gru": "roberta_gru.yaml",
        "roberta_bilstm": "roberta_bilstm.yaml",
        "roberta_tcn": "roberta_tcn.yaml",
        "roberta_tcn_attn": "roberta_tcn_attn.yaml",
        "roberta_tcn_attention": "roberta_tcn_attn.yaml",
        "roberta_tcn_attn_no_residual": "roberta_tcn_attn_no_residual.yaml",
        "roberta_tcn_attn_residual": "roberta_tcn_attn_residual.yaml",
    }

    with dataset_cfg_path.open("r", encoding="utf-8") as fh:
        cfg["dataset"] = yaml.safe_load(fh) or {}

    with (REPO_ROOT / "configs" / "models" / model_cfg_map[variant]).open("r", encoding="utf-8") as fh:
        cfg["model"] = yaml.safe_load(fh) or {}

    cfg.setdefault("training", {})
    cfg.setdefault("preprocessing", {})
    cfg.setdefault("runtime", {})
    cfg.setdefault("dataset", {})
    cfg.setdefault("model", {})
    cfg.setdefault("data_cache", {})
    cfg["dataset"].setdefault("cache", {})

    # Apply top-level default cache policy into dataset cache.
    cfg["dataset"]["cache"] = deep_merge(dict(cfg.get("data_cache", {})), dict(cfg["dataset"].get("cache", {})))

    cfg["seed"] = int(seed)
    cfg["runtime"]["device"] = default_device(device)
    cfg["dataset"]["sample_mode"] = bool(sample_mode)
    cfg["dataset"]["sample_size"] = int(sample_size)
    cfg["model"]["variant"] = variant

    if hidden_units is not None:
        cfg["model"]["hidden_units"] = int(hidden_units)
    if learning_rate is not None:
        cfg["training"]["learning_rate"] = float(learning_rate)
    if max_seq_len is not None:
        cfg["preprocessing"]["max_seq_len"] = int(max_seq_len)
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    if batch_size is not None:
        cfg["training"]["batch_size"] = int(batch_size)

    if cache_root is not None:
        cfg["dataset"]["cache"]["root"] = str(cache_root)
    if build_cache_if_missing is not None:
        cfg["dataset"]["cache"]["build_if_missing"] = bool(build_cache_if_missing)
    if cache_enabled is not None:
        cfg["dataset"]["cache"]["enabled"] = bool(cache_enabled)

    if model_overrides:
        cfg["model"] = deep_merge(cfg["model"], model_overrides)

    return cfg


def _prepare_model_inputs(cfg: Dict[str, Any], data_bundle: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = deepcopy(cfg["model"])
    preprocess_cfg = cfg["preprocessing"]
    dataset_cfg = cfg["dataset"]

    model_cfg["vocab_size"] = int(data_bundle["vocab_size"])
    model_cfg["encoder_name"] = str(preprocess_cfg.get("tokenizer_name", "roberta-base"))
    model_cfg["max_seq_len"] = int(preprocess_cfg.get("max_seq_len", 512))

    sample_mode = bool(dataset_cfg.get("sample_mode", False))
    model_cfg["local_files_only"] = sample_mode
    model_cfg["allow_random_init"] = sample_mode
    return model_cfg


def run_train_eval(cfg: Dict[str, Any], *, dry_run: bool = False, eval_split: str = "test") -> Dict[str, Any]:
    set_seed(int(cfg.get("seed", 42)))

    training_cfg = cfg["training"]
    preprocess_cfg = cfg["preprocessing"]
    dataset_cfg = cfg["dataset"]

    device = cfg["runtime"].get("device") or default_device()
    batch_size = int(training_cfg.get("batch_size", 8))
    epochs = int(training_cfg.get("epochs", 5))
    learning_rate = float(training_cfg.get("learning_rate", 1e-5))

    data_bundle = build_data_bundle(dataset_cfg, preprocess_cfg, training_cfg, seed=int(cfg["seed"]))
    model_cfg = _prepare_model_inputs(cfg, data_bundle)

    model = create_model(model_cfg, num_labels=int(data_bundle["num_labels"])).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    train_split = data_bundle["train"]
    n_train = train_split["labels"].shape[0]
    max_steps = 1 if dry_run else max(1, math.ceil(n_train / batch_size))

    timer = ThroughputTimer()
    timer.start()

    model.train()
    for _ in range(epochs):
        for step in range(max_steps):
            start = step * batch_size
            end = min(start + batch_size, n_train)

            input_ids = train_split["input_ids"][start:end].to(device)
            attention_mask = train_split["attention_mask"][start:end].to(device)
            labels = train_split["labels"][start:end].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if dry_run:
                break

        if dry_run:
            break

    train_steps_per_sec = timer.stop(max_steps)

    split = data_bundle[eval_split]
    preds: List[int] = []
    labels_all: List[int] = []

    model.eval()
    with torch.inference_mode():
        for start in range(0, split["labels"].shape[0], batch_size):
            end = min(start + batch_size, split["labels"].shape[0])

            input_ids = split["input_ids"][start:end].to(device)
            attention_mask = split["attention_mask"][start:end].to(device)
            labels = split["labels"][start:end]

            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    return {
        "accuracy": float(accuracy(preds, labels_all)),
        "macro_f1": float(macro_f1(preds, labels_all)),
        "train_steps_per_sec": float(train_steps_per_sec),
        "variant": str(model_cfg.get("variant")),
        "source": str(data_bundle.get("source", "")),
        "raw_source": str(data_bundle.get("raw_source", "")),
        "cache_dir": str(data_bundle.get("cache_dir", "")),
    }


def measure_validation_throughput(cfg: Dict[str, Any], *, split: str = "val", warmup_steps: int = 10) -> float:
    set_seed(int(cfg.get("seed", 42)))

    training_cfg = cfg["training"]
    preprocess_cfg = cfg["preprocessing"]
    dataset_cfg = cfg["dataset"]

    device = cfg["runtime"].get("device") or default_device()
    batch_size = int(training_cfg.get("batch_size", 8))

    data_bundle = build_data_bundle(dataset_cfg, preprocess_cfg, training_cfg, seed=int(cfg["seed"]))
    model_cfg = _prepare_model_inputs(cfg, data_bundle)
    model = create_model(model_cfg, num_labels=int(data_bundle["num_labels"])).to(device)
    model.eval()

    target = data_bundle[split]
    num_items = int(target["labels"].shape[0])
    if num_items == 0:
        return 0.0

    steps = max(1, math.ceil(num_items / batch_size))

    with torch.inference_mode():
        for step in range(min(warmup_steps, steps)):
            start = step * batch_size
            end = min(start + batch_size, num_items)
            input_ids = target["input_ids"][start:end].to(device)
            attention_mask = target["attention_mask"][start:end].to(device)
            _ = model(input_ids, attention_mask)

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for step in range(steps):
            start = step * batch_size
            end = min(start + batch_size, num_items)
            input_ids = target["input_ids"][start:end].to(device)
            attention_mask = target["attention_mask"][start:end].to(device)
            _ = model(input_ids, attention_mask)

        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize()

        elapsed = time.perf_counter() - t0

    return float(steps / elapsed) if elapsed > 0 else 0.0


def write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))