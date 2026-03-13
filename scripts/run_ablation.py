from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from scripts._experiment_utils import (
    DATASET_KEYS,
    TABLE1_MODELS,
    compose_runtime_config,
    format_float,
    run_train_eval,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ablation-style benchmark with cache-aware data loading.")
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


def main() -> None:
    args = parse_args()

    rows: List[List[str]] = []
    for dataset in args.datasets:
        for display_name, variant in TABLE1_MODELS:
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
            rows.append(
                [
                    dataset,
                    display_name,
                    format_float(result["accuracy"]),
                    format_float(result["macro_f1"]),
                    format_float(result["train_steps_per_sec"]),
                    result.get("cache_dir", ""),
                ]
            )
            print(
                f"[{dataset}] {display_name} | "
                f"acc={result['accuracy']:.4f} "
                f"f1={result['macro_f1']:.4f} "
                f"steps/s={result['train_steps_per_sec']:.4f}"
            )

    write_csv(
        Path(args.output),
        header=["Dataset", "Model", "Accuracy", "Macro-F1", "TrainSteps/s", "CacheDir"],
        rows=rows,
    )
    print(f"[OK] wrote: {args.output}")


if __name__ == "__main__":
    main()