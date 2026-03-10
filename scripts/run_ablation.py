#!/usr/bin/env python3
"""Run Table 2 ablation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from _experiment_utils import DATASET_KEYS, compose_runtime_config, run_train_eval, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 2 ablation experiments.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output", type=str, default="outputs/raw/table2_raw.csv")
    parser.add_argument("--hidden-units", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-6)
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
    ablations = [
        ("TCN", "roberta_tcn", {}),
        ("TCN-Attnw/o Residual", "roberta_tcn_attn", {"use_residual_fusion": False}),
        ("TCN-Attnw/ Residual", "roberta_tcn_attn", {"use_residual_fusion": True}),
    ]

    rows = []
    for model_name, variant, overrides in ablations:
        metrics = {}
        for dataset in DATASET_KEYS:
            cfg = compose_runtime_config(
                config_path=args.config,
                dataset=dataset,
                variant=variant,
                hidden_units=args.hidden_units,
                learning_rate=args.lr,
                max_seq_len=args.max_seq_len,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_mode=args.sample_mode,
                sample_size=args.sample_size,
                seed=args.seed,
                device=args.device,
                model_overrides=overrides,
            )
            result = run_train_eval(cfg, dry_run=args.dry_run, eval_split="test")
            metrics[dataset] = result
            print(f"[table2] model={model_name} dataset={dataset} acc={result['accuracy']:.4f} f1={result['macro_f1']:.4f}")
        rows.append(
            [
                model_name,
                metrics["imdb"]["accuracy"],
                metrics["imdb"]["macro_f1"],
                metrics["twitter_us_airline"]["accuracy"],
                metrics["twitter_us_airline"]["macro_f1"],
                metrics["sentiment140"]["accuracy"],
                metrics["sentiment140"]["macro_f1"],
            ]
        )

    header = ["Model", "IMDb_Acc", "IMDb_F1", "Twitter_Acc", "Twitter_F1", "Sentiment140_Acc", "Sentiment140_F1"]
    write_csv(Path(args.output), header, rows)
    print(f"[done] wrote raw table2 results: {args.output}")


if __name__ == "__main__":
    main()
