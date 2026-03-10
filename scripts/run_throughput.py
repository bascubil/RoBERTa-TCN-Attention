from __future__ import annotations

import argparse
from pathlib import Path

from _experiment_utils import DATASET_KEYS, compose_runtime_config, measure_validation_throughput, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Table 3 throughput benchmarks.")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument("--output", type=str, default="outputs/raw/table3_raw.csv")
    parser.add_argument("--hidden-units", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None)
    parser.add_argument("--sample-mode", action="store_true")
    parser.add_argument("--sample-size", type=int, default=64)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models = [
        ("RoBERTa-LSTM", "roberta_lstm", {}),
        ("RoBERTa-BiLSTM", "roberta_bilstm", {}),
        ("RoBERTa-GRU", "roberta_gru", {}),
        ("RoBERTa-TCN", "roberta_tcn", {}),
        ("RoBERTa-TCN-Attn", "roberta_tcn_attn", {"use_residual_fusion": True}),
    ]

    rows = []
    for model_name, variant, overrides in models:
        row = [model_name]
        for dataset in DATASET_KEYS:
            cfg = compose_runtime_config(
                config_path=args.config,
                dataset=dataset,
                variant=variant,
                hidden_units=args.hidden_units,
                learning_rate=None,
                max_seq_len=args.max_seq_len,
                epochs=None,
                batch_size=args.batch_size,
                sample_mode=args.sample_mode,
                sample_size=args.sample_size,
                seed=args.seed,
                device=args.device,
                model_overrides=overrides,
            )
            sps = measure_validation_throughput(cfg, split="val", warmup_steps=args.warmup_steps)
            row.append(sps)
            print(f"[table3] model={model_name} dataset={dataset} steps_per_sec={sps:.4f}")
        rows.append(row)

    header = ["Model", "IMDb", "Twitter", "Sentiment140"]
    write_csv(Path(args.output), header, rows)
    print(f"[done] wrote raw table3 results: {args.output}")


if __name__ == "__main__":
    main()
