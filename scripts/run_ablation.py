from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._experiment_utils import (
    DATASET_KEYS,
    TABLE2_MODELS,
    compose_runtime_config,
    format_float,
    run_train_eval,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 2 ablation benchmark.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_KEYS),
        default=list(DATASET_KEYS),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--build-cache-if-missing", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/raw/table2_raw.csv")
    return parser.parse_args()


def _cell(metrics_by_dataset: Dict[str, Dict[str, float]], dataset: str, key: str) -> str:
    if dataset not in metrics_by_dataset:
        return ""
    return format_float(metrics_by_dataset[dataset][key])


def main() -> None:
    args = parse_args()
    rows: List[List[str]] = []

    for display_name, variant in TABLE2_MODELS:
        metrics_by_dataset: Dict[str, Dict[str, float]] = {}

        for dataset in args.datasets:
            cfg = compose_runtime_config(
                config_path=args.config,
                dataset=dataset,
                variant=variant,
                hidden_units=None,
                learning_rate=None,
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
            result = run_train_eval(cfg)
            metrics_by_dataset[dataset] = result

            print(
                f"[{dataset}] {display_name} | "
                f"acc={result['accuracy']:.4f} "
                f"f1={result['macro_f1']:.4f}"
            )

        rows.append([
            display_name,
            _cell(metrics_by_dataset, "imdb", "accuracy"),
            _cell(metrics_by_dataset, "imdb", "macro_f1"),
            _cell(metrics_by_dataset, "twitter_us_airline", "accuracy"),
            _cell(metrics_by_dataset, "twitter_us_airline", "macro_f1"),
            _cell(metrics_by_dataset, "sentiment140", "accuracy"),
            _cell(metrics_by_dataset, "sentiment140", "macro_f1"),
        ])

    write_csv(
        Path(args.output),
        header=["Model", "IMDb_Acc", "IMDb_F1", "Twitter_Acc", "Twitter_F1", "Sentiment140_Acc", "Sentiment140_F1",],
        rows=rows,
    )
    print(f"[OK] wrote: {args.output}")


if __name__ == "__main__":
    main()