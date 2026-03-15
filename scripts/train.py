from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import argparse
import json

from scripts._experiment_utils import compose_runtime_config, run_train_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval one model with cache-aware loading.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--dataset", type=str, required=True, choices=["imdb", "twitter_us_airline", "sentiment140"])
    parser.add_argument(
        "--variant", "--model-variant",
        dest = "variant",
        type=str,
        required=True,
        choices=["roberta_base", "roberta_lstm", "roberta_bilstm", "roberta_gru", "roberta_tcn", "roberta_tcn_attn", "roberta_tcn_attn_no_residual","roberta_tcn_attn_residual"],
    )
    parser.add_argument("--hidden-units", type=int, default=None)
    parser.add_argument("--learning-rate","--lr", dest = "learning_rate",type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--build-cache-if-missing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = compose_runtime_config(
        config_path=args.config,
        dataset=args.dataset,
        variant=args.variant,
        hidden_units=args.hidden_units,
        learning_rate=args.learning_rate,
        max_seq_len=args.max_seq_len,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_mode=False,
        sample_size=64,
        seed=args.seed,
        device=args.device,
        cache_root=args.cache_root,
        build_cache_if_missing=args.build_cache_if_missing,
        cache_enabled=True,
    )
    result = run_train_eval(cfg, dry_run=bool(args.dry_run))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()