# RoBERTa-TCN-Attention (Neurocomputing submission code)

This repository contains the companion code for the manuscript:

> A Sentiment Classification Model Using RoBERTa-TCN with Self-Attention
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

## Environment

- Python 3.10+
- PyTorch
- Hugging Face `transformers`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data setup (raw data is not included)

This repo does not redistribute the raw datasets.
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

- Throughput numbers (steps/sec) depend on hardware/software and may differ from the paper unless you run on a comparable environment.
- The hidden-unit grid and learning-rate grid used for Table 1 are defined directly in configs/experiment.yaml.

## License

MIT License (see LICENSE).

