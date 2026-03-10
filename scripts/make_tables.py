#!/usr/bin/env python3
"""Render Table 1/2/3 CSVs in the manuscript layout."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

from _experiment_utils import format_float


TABLE1_ORDER = [
    ("RoBERTa-base", "-"),
    ("RoBERTa-LSTM", "128"),
    ("RoBERTa-LSTM", "256"),
    ("RoBERTa-LSTM", "512"),
    ("RoBERTa-BiLSTM", "128"),
    ("RoBERTa-BiLSTM", "256"),
    ("RoBERTa-BiLSTM", "512"),
    ("RoBERTa-GRU", "128"),
    ("RoBERTa-GRU", "256"),
    ("RoBERTa-GRU", "512"),
    ("RoBERTa-TCN", "128"),
    ("RoBERTa-TCN", "256"),
    ("RoBERTa-TCN", "512"),
    ("RoBERTa-TCN-Attention", "128"),
    ("RoBERTa-TCN-Attention", "256"),
    ("RoBERTa-TCN-Attention", "512"),
]
LR_ORDER = ["1e-4", "1e-5", "1e-6"]
TABLE2_ORDER = ["TCN", "TCN-Attnw/o Residual", "TCN-Attnw/ Residual"]
TABLE3_ORDER = ["RoBERTa-LSTM", "RoBERTa-BiLSTM", "RoBERTa-GRU", "RoBERTa-TCN", "RoBERTa-TCN-Attn"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-layout tables from raw experiment CSVs.")
    parser.add_argument("--table1-raw", type=str, default="outputs/raw/table1_raw.csv")
    parser.add_argument("--table2-raw", type=str, default="outputs/raw/table2_raw.csv")
    parser.add_argument("--table3-raw", type=str, default="outputs/raw/table3_raw.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/tables")
    return parser.parse_args()


def _read_dict_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def build_table1(raw_path: Path, out_path: Path) -> None:
    rows = _read_dict_rows(raw_path)
    lookup: Dict[Tuple[str, str, str], Dict[str, str]] = {}
    for row in rows:
        key = (row["Model"], str(row["Hidden units"]), row["LR"])
        lookup[key] = row

    out_rows = [
        ["Model", "Hidden units", "LR", "IMDb", "IMDb", "Twitter", "Twitter", "Sentiment140", "Sentiment140"],
        ["Model", "Hidden units", "LR", "Acc", "F1", "Acc", "F1", "Acc", "F1"],
    ]
    for model, hidden in TABLE1_ORDER:
        for lr in LR_ORDER:
            src = lookup.get((model, hidden, lr), {})
            out_rows.append(
                [
                    model,
                    hidden,
                    lr,
                    format_float(src.get("IMDb_Acc", 0.0)),
                    format_float(src.get("IMDb_F1", 0.0)),
                    format_float(src.get("Twitter_Acc", 0.0)),
                    format_float(src.get("Twitter_F1", 0.0)),
                    format_float(src.get("Sentiment140_Acc", 0.0)),
                    format_float(src.get("Sentiment140_F1", 0.0)),
                ]
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(out_rows)


def build_table2(raw_path: Path, out_path: Path) -> None:
    rows = _read_dict_rows(raw_path)
    lookup = {row["Model"]: row for row in rows}
    out_rows = [
        ["Model", "IMDb", "IMDb", "Twitter", "Twitter", "Sentiment140", "Sentiment140"],
        ["Model", "Acc", "F1", "Acc", "F1", "Acc", "F1"],
    ]
    for model in TABLE2_ORDER:
        src = lookup.get(model, {})
        out_rows.append(
            [
                model,
                format_float(src.get("IMDb_Acc", 0.0)),
                format_float(src.get("IMDb_F1", 0.0)),
                format_float(src.get("Twitter_Acc", 0.0)),
                format_float(src.get("Twitter_F1", 0.0)),
                format_float(src.get("Sentiment140_Acc", 0.0)),
                format_float(src.get("Sentiment140_F1", 0.0)),
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(out_rows)


def build_table3(raw_path: Path, out_path: Path) -> None:
    rows = _read_dict_rows(raw_path)
    lookup = {row["Model"]: row for row in rows}
    out_rows = [["Model", "IMDb", "Twitter", "Sentiment140"]]
    for model in TABLE3_ORDER:
        src = lookup.get(model, {})
        out_rows.append(
            [
                model,
                format_float(src.get("IMDb", 0.0)),
                format_float(src.get("Twitter", 0.0)),
                format_float(src.get("Sentiment140", 0.0)),
            ]
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerows(out_rows)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    table1_out = out_dir / "table1.csv"
    table2_out = out_dir / "table2.csv"
    table3_out = out_dir / "table3.csv"
    build_table1(Path(args.table1_raw), table1_out)
    build_table2(Path(args.table2_raw), table2_out)
    build_table3(Path(args.table3_raw), table3_out)
    print(f"[done] wrote {table1_out}, {table2_out}, {table3_out}")


if __name__ == "__main__":
    main()
