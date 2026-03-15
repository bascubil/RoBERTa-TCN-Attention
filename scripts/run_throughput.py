from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts._experiment_utils import (
    DATASET_DISPLAY,
    DATASET_KEYS,
    TABLE1_MODELS,
    compose_runtime_config,
    format_float,
    measure_validation_throughput,
    write_csv,
)

import torch
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure validation throughput on the selected datasets")
    parser.add_argument("--config", type=str, default="configs/experiment.yaml")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_KEYS),
        default=list(DATASET_KEYS),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--cache-root", type=str, default=None)
    parser.add_argument("--build-cache-if-missing", action="store_true")
    parser.add_argument("--output", type=str, default="outputs/raw/table3_raw.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: List[List[str]] = []
    for model_name, variant in TABLE1_MODELS:
    # for model_name, variant in [("RoBERTa-TCN", "roberta_tcn")]:
        per_dataset: Dict[str, str] = {}
        for dataset in args.datasets:
            cfg = compose_runtime_config(
                config_path=args.config,
                dataset=dataset,
                variant=variant,
                hidden_units=None,
                learning_rate=None,
                max_seq_len=args.max_seq_len,
                epochs=None,
                batch_size=args.batch_size,
                sample_mode=False,
                sample_size=64,
                seed=args.seed,
                device=args.device,
                cache_root=args.cache_root,
                build_cache_if_missing=args.build_cache_if_missing,
                cache_enabled=True,
            )
            throughput = measure_validation_throughput(cfg, split="val")
            per_dataset[dataset] = format_float(throughput)
            print(f"[{dataset}] {model_name}: {throughput:.4f} steps/s")

        row = [model_name]
        for dataset in args.datasets:
            row.append(per_dataset.get(dataset, ""))
        rows.append(row)

    header = ["Model"] + [DATASET_DISPLAY[d] for d in args.datasets]
    write_csv(Path(args.output), header=header, rows=rows)
    print(f"[OK] wrote: {args.output}")


if __name__ == "__main__":
    main()