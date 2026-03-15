from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data import prepare_dataset_cache
from src.utils.seed import set_seed

from scripts._experiment_utils import (
    DATASET_KEYS,
    compose_runtime_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="token cache for datasets.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiment.yaml",
        help="Experiment config path.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=list(DATASET_KEYS),
        choices=list(DATASET_KEYS),
        help="Dataset keys to cache.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild the cache even if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    for dataset in args.datasets:
        cfg = compose_runtime_config(
            config_path=args.config,
            dataset=dataset,
            variant="roberta_base",
            hidden_units=None,
            learning_rate=None,
            max_seq_len=args.max_seq_len,
            epochs=None,
            batch_size=None,
            sample_mode=False,
            sample_size=64,
            seed=args.seed,
            device=None,
            cache_root=args.cache_root,
            build_cache_if_missing=True,
            cache_enabled=True,
            model_overrides=None,
        )

        cache_dir = prepare_dataset_cache(
            dataset_cfg=cfg["dataset"],
            preprocessing_cfg=cfg["preprocessing"],
            training_cfg=cfg["training"],
            seed=int(cfg["seed"]),
            force_rebuild=bool(args.force_rebuild),
        )
        print(f"[OK] {dataset}: {cache_dir}")


if __name__ == "__main__":
    main()