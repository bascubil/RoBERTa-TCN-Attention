#!/usr/bin/env python3
"""Run Table 1 hidden_units x lr grid across models and datasets."""

from __future__ import annotations

import argparse
from pathlib import Path

from _experiment_utils import (
    DATASET_KEYS,
    TABLE1_MODELS,
    canonical_hidden,
    compose_runtime_config,
    format_lr,
    load_grid_settings,
    run_train_eval,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 1 experiment grid.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output", type=str, default="outputs/raw/table1_raw.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hidden_grid, lr_grid = load_grid_settings(args.config)

    rows = []
    for model_name, variant in TABLE1_MODELS:
        hidden_values = [None] if variant == "roberta_base" else hidden_grid
        for hidden in hidden_values:
            for lr in lr_grid:
                metrics_by_dataset = {}
                for dataset in DATASET_KEYS:
                    cfg = compose_runtime_config(
                        config_path=args.config,
                        dataset=dataset,
                        variant=variant,
                        hidden_units=hidden,
                        learning_rate=lr,
                        max_seq_len=args.max_seq_len,
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        sample_mode=args.sample_mode,
                        sample_size=args.sample_size,
                        seed=args.seed,
                        device=args.device,
                    )
                    result = run_train_eval(cfg, dry_run=args.dry_run, eval_split="test")
                    metrics_by_dataset[dataset] = result
                    print(
                        f"[table1] model={model_name} hidden={canonical_hidden(hidden)} lr={format_lr(lr)} "
                        f"dataset={dataset} acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f}"
                    )

                rows.append(
                    [
                        model_name,
                        canonical_hidden(hidden),
                        format_lr(lr),
                        metrics_by_dataset["imdb"]["accuracy"],
                        metrics_by_dataset["imdb"]["macro_f1"],
                        metrics_by_dataset["twitter_us_airline"]["accuracy"],
                        metrics_by_dataset["twitter_us_airline"]["macro_f1"],
                        metrics_by_dataset["sentiment140"]["accuracy"],
                        metrics_by_dataset["sentiment140"]["macro_f1"],
                    ]
                )

    header = ["Model", "Hidden units", "LR", "IMDb_Acc", "IMDb_F1", "Twitter_Acc", "Twitter_F1", "Sentiment140_Acc", "Sentiment140_F1"]
    write_csv(Path(args.output), header, rows)
    print(f"[done] wrote raw table1 results: {args.output}")


if __name__ == "__main__":
    main()
