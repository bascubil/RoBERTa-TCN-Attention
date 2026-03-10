from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import build_data_bundle
from src.metrics import accuracy, macro_f1
from src.models import SUPPORTED_VARIANTS, create_model
from src.utils.config import deep_merge, load_composed_config, load_yaml, resolve_config_path
from src.utils.seed import set_seed


MODEL_FILE_BY_VARIANT = {
    "roberta_base": "roberta_base.yaml",
    "roberta_lstm": "roberta_lstm.yaml",
    "roberta_gru": "roberta_gru.yaml",
    "roberta_bilstm": "roberta_bilstm.yaml",
    "roberta_tcn": "roberta_tcn.yaml",
    "roberta_tcn_attn": "roberta_tcn_attn.yaml",
    "roberta_tcn_attention": "roberta_tcn_attn.yaml",
}


def _normalize_variant(name: str) -> str:
    return name.strip().lower().replace("-", "_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one model variant with one command.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml", help="Experiment YAML path.")
    parser.add_argument("--dataset", type=str, choices=["imdb", "twitter_us_airline", "sentiment140"], help="Dataset template name.")
    parser.add_argument("--dataset-config", type=str, help="Path to dataset YAML.")
    parser.add_argument("--dataset-path", type=str, help="Path override for the dataset CSV file.")
    parser.add_argument("--sample-mode", action="store_true", help="Use small built-in sample records instead of full CSV.")
    parser.add_argument("--sample-size", type=int, help="Number of sample records to synthesize when sample mode is on.")
    parser.add_argument("--model-variant", type=str, help="Model variant.")
    parser.add_argument("--model-config", type=str, help="Path to model YAML.")
    parser.add_argument("--batch-size", type=int, help="Override batch_size.")
    parser.add_argument("--max-seq-len", type=int, help="Override max_seq_len.")
    parser.add_argument("--seed", type=int, help="Override random seed.")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], help="Runtime device.")
    parser.add_argument("--checkpoint", type=str, help="Optional checkpoint path.")
    parser.add_argument("--split", type=str, choices=["train", "val", "test"], default="test")
    return parser.parse_args()


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    merged = deep_merge({}, cfg)
    if args.dataset:
        merged["dataset"] = load_yaml(REPO_ROOT / "configs" / "datasets" / f"{args.dataset}.yaml")
    if args.dataset_config:
        path = resolve_config_path(args.dataset_config, base_dir=REPO_ROOT, repo_root=REPO_ROOT)
        merged["dataset"] = load_yaml(path)

    if args.model_variant:
        variant = _normalize_variant(args.model_variant)
        if variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unsupported --model-variant '{args.model_variant}'. Use one of: {SUPPORTED_VARIANTS}")
        filename = MODEL_FILE_BY_VARIANT.get(variant)
        if filename is None:
            raise ValueError(f"No model template for variant '{variant}'")
        merged["model"] = load_yaml(REPO_ROOT / "configs" / "models" / filename)
        merged["model"]["variant"] = variant
    if args.model_config:
        path = resolve_config_path(args.model_config, base_dir=REPO_ROOT, repo_root=REPO_ROOT)
        merged["model"] = deep_merge(merged.get("model", {}), load_yaml(path))

    merged.setdefault("training", {})
    merged.setdefault("preprocessing", {})
    merged.setdefault("runtime", {})
    merged.setdefault("model", {})
    merged.setdefault("dataset", {})
    if args.dataset_path is not None:
        merged["dataset"]["path"] = args.dataset_path
    if args.sample_mode:
        merged["dataset"]["sample_mode"] = True
    if args.sample_size is not None:
        merged["dataset"]["sample_size"] = int(args.sample_size)

    if args.batch_size is not None:
        merged["training"]["batch_size"] = int(args.batch_size)
    if args.max_seq_len is not None:
        merged["preprocessing"]["max_seq_len"] = int(args.max_seq_len)
    if args.seed is not None:
        merged["seed"] = int(args.seed)
    if args.device is not None:
        merged["runtime"]["device"] = args.device
    return merged


def main() -> None:
    args = parse_args()
    exp_path = resolve_config_path(args.config, base_dir=REPO_ROOT, repo_root=REPO_ROOT)
    cfg = load_composed_config(exp_path, repo_root=REPO_ROOT)
    cfg = apply_cli_overrides(cfg, args)

    seed = int(cfg.get("seed", 42))
    set_seed(seed)

    training_cfg = cfg["training"]
    preprocess_cfg = cfg["preprocessing"]
    dataset_cfg = cfg["dataset"]
    model_cfg = cfg["model"]
    runtime_cfg = cfg["runtime"]

    device = runtime_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = int(training_cfg.get("batch_size", 8))

    data_bundle = build_data_bundle(dataset_cfg, preprocess_cfg, training_cfg, seed=seed)
    model_cfg["vocab_size"] = int(data_bundle["vocab_size"])
    model_cfg["encoder_name"] = str(preprocess_cfg.get("tokenizer_name", "roberta-base"))
    model_cfg["max_seq_len"] = int(preprocess_cfg.get("max_seq_len", 512))
    sample_mode = bool(dataset_cfg.get("sample_mode", False))
    model_cfg["local_files_only"] = sample_mode
    model_cfg["allow_random_init"] = sample_mode
    model = create_model(model_cfg, num_labels=int(data_bundle["num_labels"])).to(device)

    if args.checkpoint:
        ckpt_path = resolve_config_path(args.checkpoint, base_dir=REPO_ROOT, repo_root=REPO_ROOT)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

    model.eval()
    split = data_bundle[args.split]
    preds = []
    labels_all = []
    with torch.no_grad():
        for start in range(0, split["labels"].shape[0], batch_size):
            end = min(start + batch_size, split["labels"].shape[0])
            input_ids = split["input_ids"][start:end].to(device)
            attention_mask = split["attention_mask"][start:end].to(device)
            labels = split["labels"][start:end]
            logits = model(input_ids, attention_mask)
            preds.extend(torch.argmax(logits, dim=-1).detach().cpu().tolist())
            labels_all.extend(labels.detach().cpu().tolist())

    acc = accuracy(preds, labels_all)
    f1 = macro_f1(preds, labels_all)
    print("evaluation_complete")
    print(f"split={args.split}")
    print(f"dataset_source={data_bundle['source']}")
    print(f"variant={model_cfg.get('variant')}")
    print(f"accuracy={acc:.6f}")
    print(f"macro_f1={f1:.6f}")


if __name__ == "__main__":
    main()
