# RoBERTa-TCN-Attention (Neurocomputing submission code)

This repository contains the companion code for the manuscript:

> **A Sentiment Classification Model Using RoBERTa-TCN with Self-Attention**  
> Namwoo Kim, Taehyun Ha

It provides a CLI to train/evaluate model variants and to reproduce the paper tables.

## What is implemented

Model variants (all share the same RoBERTa encoder backbone):

- `roberta_base`
- `roberta_lstm`
- `roberta_gru`
- `roberta_bilstm`
- `roberta_tcn`
- `roberta_tcn_attn` (RoBERTa-TCN-Attention)

### TCN implementation details (matches manuscript)

The TCN head is implemented as a stack of residual blocks using:

- **Dilated causal 1D convolutions** (left padding only; no future token leakage)
- **Exponentially increasing dilation factors**: `1, 2, 4, ...`
- Residual blocks with `(Conv → Norm → ReLU → Dropout) × 2`
- Sequence length is preserved, so the upstream **attention mask is reused**.

### RNN baseline behavior

The RNN baselines are kept in the paper-aligned form used in the original experiments.
The repository no longer exposes an alternate pooling mode for these baselines.

## Environment

- Python 3.10+
- PyTorch
- Hugging Face `transformers`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data setup (raw data is not included)

This repo **does not** redistribute the raw datasets.
Place the downloaded CSV files under `data/raw/`:

- `data/raw/IMDB Dataset.csv`
- `data/raw/Twitter US Airline Sentiment Dataset.csv`
- `data/raw/Sentiment140 Dataset.csv`

See `data/README.md` for required column names.

## Training / evaluation

Base experiment config:

- `configs/experiment.yaml`

Train one variant (example: RoBERTa-TCN-Attention on IMDb):

```bash
python scripts/train.py \
  --config configs/experiment.yaml \
  --dataset imdb \
  --model-variant roberta_tcn_attn \
  --hidden-units 256 \
  --lr 1e-5
```

Evaluate a saved checkpoint:

```bash
python scripts/eval.py \
  --config configs/experiment.yaml \
  --dataset imdb \
  --model-variant roberta_tcn_attn \
  --checkpoint outputs/last_model.pt
```

## Reproducing Tables 1–3

Run the table experiments (raw results):

```bash
python scripts/run_grid.py --config configs/experiment.yaml
python scripts/run_ablation.py --config configs/experiment.yaml
python scripts/run_throughput.py --config configs/experiment.yaml
```

Build the final table CSVs:

```bash
python scripts/make_tables.py
```

Notes:

- Throughput numbers (`steps/sec`) depend on hardware/software and may differ from the paper unless you run on a comparable environment.
- The hidden-unit grid and learning-rate grid used for Table 1 are defined directly in `configs/experiment.yaml`.

## Sample mode (no datasets needed)

For quick smoke tests without downloading datasets:

```bash
python scripts/run_grid.py --sample-mode --dry-run --sample-size 24 --max-seq-len 64 --epochs 1
python scripts/run_ablation.py --sample-mode --dry-run --sample-size 24 --max-seq-len 64 --epochs 1
python scripts/run_throughput.py --sample-mode --sample-size 24 --max-seq-len 64
python scripts/make_tables.py
```

Sample mode uses a small built-in sample dataset and (if pretrained weights are unavailable) can fall back to a lightweight randomly initialized RoBERTa for pipeline validation.

## License

MIT License (see `LICENSE`).

## Citation

If you use this repository, please cite the manuscript and/or this code (see `CITATION.cff`).
