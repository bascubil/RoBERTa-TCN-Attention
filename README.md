# RoBERTa-TCN-Attention

This repository contains the code for the manuscript:

> A Sentiment Classification Model Using RoBERTa-TCN with Self-Attention  
> Namwoo Kim, Taehyun Ha

It includes:
- preprocess and token caching
- train and evaluate several model variants
- generate the paper tables

## Implemented model variants

All variants use the same RoBERTa encoder backbone and differ in the classification head.

- `roberta_base` — RoBERTa-base
- `roberta_lstm` — RoBERTa-LSTM
- `roberta_gru` — RoBERTa-GRU
- `roberta_bilstm` — RoBERTa-BiLSTM
- `roberta_tcn` — RoBERTa-TCN
- `roberta_tcn_attn` — RoBERTa-TCN-Attention
- `roberta_tcn_attn_no_residual` — Table 2 ablation: TCN-Attn without residual connection
- `roberta_tcn_attn_residual` — Table 2 ablation: TCN-Attn with residual connection

## Repository layout

```text
.
├── configs/
│   ├── experiment.yaml
│   ├── datasets/
│   └── models/
├── data/
│   ├── raw/
│   ├── cache/
│   └── README.md
├── scripts/
│   ├── prepare_dataset_cache.py
│   ├── train.py
│   ├── eval.py
│   ├── run_grid.py
│   ├── run_ablation.py
│   ├── run_throughput.py
│   └── make_tables.py
├── src/
│   ├── data/
│   ├── metrics/
│   ├── models/
│   └── utils/
├── requirements.txt
└── README.md
```

## Environment

- Python 3.10+
- Main dependencies are pinned in `requirements.txt`:
  - `torch==2.8.0`
  - `transformers==4.57.3`
  - `pandas==2.2.3`
  - `PyYAML==6.0.2`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data setup

This repository does not redistribute the raw datasets used in the paper.  
Please download the datasets from their original sources and place the files under `data/raw/` with the exact filenames below:

- `data/raw/IMDB Dataset.csv`
- `data/raw/Twitter US Airline Sentiment Dataset.csv`
- `data/raw/Sentiment140 Dataset.csv`

Expected columns:

- IMDb: `review`, `sentiment`
  - label values: `positive`, `negative`
- Twitter US Airline Sentiment: `text`, `airline_sentiment`
  - label values: `positive`, `neutral`, `negative`
- Sentiment140:
  - either the original 6-column CSV  
    (`target,id,date,query,user,text`)
  - or an equivalent CSV containing at least `target` and `text`

Dataset sources:

- IMDb Review Dataset
  - original source: https://ai.stanford.edu/~amaas/data/sentiment/
  - Kaggle mirror: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
- Twitter US Airline Sentiment
  - Kaggle page: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- Sentiment140
  - original download: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

Preprocessing in this codebase removes:
- URLs
- `@mentions`

See `data/README.md` for the same dataset notes in a compact form.

## Default experiment configuration

The base experiment config is `configs/experiment.yaml`.

Default settings:
- `seed: 42`
- tokenizer: `roberta-base`
- `max_seq_len: 512`
- `batch_size: 8`
- `epochs: 5`
- `learning_rate: 1e-5`

Token caching is enabled by default:
- cache root: `data/cache`
- cache version: `token-cache-v1`
- `build_if_missing: false`

Because `build_if_missing` is disabled by default, you should prepare the token cache before running training/evaluation scripts.

## Quick start

Run all commands from the repository root.

### 1) Prepare token cache

```bash
python scripts/prepare_dataset_cache.py \
  --config configs/experiment.yaml \
  --datasets imdb twitter_us_airline sentiment140
```

You can force a rebuild of matching caches with:

```bash
python scripts/prepare_dataset_cache.py \
  --config configs/experiment.yaml \
  --datasets imdb twitter_us_airline sentiment140 \
  --force-rebuild
```

### 2) Train one variant

Example: train RoBERTa-TCN-Attention on IMDb.

```bash
python scripts/train.py \
  --config configs/experiment.yaml \
  --dataset imdb \
  --model-variant roberta_tcn_attn \
  --hidden-units 256 \
  --lr 1e-5
```

Options:
- `--epochs`
- `--batch-size`
- `--max-seq-len`
- `--device`
- `--cache-root`
- `--build-cache-if-missing`
- `--dry-run`

### 3) Evaluate one split

`eval.py` uses the shared training/evaluation pipeline and reports metrics on the selected split.  
It does not load a saved checkpoint from disk.

Example:

```bash
python scripts/eval.py \
  --config configs/experiment.yaml \
  --dataset imdb \
  --model-variant roberta_tcn_attn \
  --epochs 1 \
  --eval-split test
```

evaluation splits:
- `train`
- `val`
- `test`

### 4) Variant reference

Paper name → CLI variant

- RoBERTa-base → `roberta_base`
- RoBERTa-LSTM → `roberta_lstm`
- RoBERTa-GRU → `roberta_gru`
- RoBERTa-BiLSTM → `roberta_bilstm`
- RoBERTa-TCN → `roberta_tcn`
- RoBERTa-TCN-Attention → `roberta_tcn_attn`
- TCN-Attn (without residual connection) → `roberta_tcn_attn_no_residual`
- TCN-Attn (with residual connection) → `roberta_tcn_attn_residual`

## Reproducing the paper tables

### Table 1: model / hidden-unit / learning-rate grid

Run the raw experiment sweep:

```bash
python scripts/run_grid.py --config configs/experiment.yaml
```

output:
- `outputs/raw/table1_raw.csv`

Notes:
- the hidden-unit grid and learning-rate grid are read from `configs/experiment.yaml`
- the script evaluates the selected models across all supported datasets
- the default grid in `configs/experiment.yaml` is:
  - hidden units: `128`, `256`, `512`
  - learning rates: `1e-4`, `1e-5`, `1e-6`

### Table 2: residual-connection ablation

Run the ablation benchmark:

```bash
python scripts/run_ablation.py --config configs/experiment.yaml
```

Default output:
- `outputs/raw/table2_raw.csv`

Ablation rows:
- `TCN`
- `TCN-Attn (without residual connection)`
- `TCN-Attn (with residual connection)`

### Table 3: throughput benchmark

Run the throughput benchmark:

```bash
python scripts/run_throughput.py --config configs/experiment.yaml
```

Default output:
- `outputs/raw/table3_raw.csv`

Notes:
- throughput is measured in steps per second
- the script measures **validation-split** throughput
- throughput depends on hardware, CUDA/cuDNN, and software versions
- on CUDA, the script enables cuDNN benchmark and TF32-related settings for measurement

### Generate the paper-layout tables

After generating the CSV files, build the table:

```bash
python scripts/make_tables.py
```

Default outputs:
- `outputs/tables/table1.csv`
- `outputs/tables/table2.csv`
- `outputs/tables/table3.csv`

You can also override the input/output paths:

```bash
python scripts/make_tables.py \
  --table1-raw outputs/raw/table1_raw.csv \
  --table2-raw outputs/raw/table2_raw.csv \
  --table3-raw outputs/raw/table3_raw.csv \
  --output-dir outputs/tables
```

## License

MIT License. See `LICENSE`.