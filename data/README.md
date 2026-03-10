# Data folder

This repository does not redistribute the raw datasets used in the paper.
These are third-party datasets distributed under their own terms, so the raw files are not included here.

Please download the datasets from their original sources and place the raw files under:

- `data/raw/IMDB Dataset.csv`
- `data/raw/Twitter US Airline Sentiment Dataset.csv`
- `data/raw/Sentiment140 Dataset.csv`

Dataset sources:
- IMDb Review Dataset
  - Original source: https://ai.stanford.edu/~amaas/data/sentiment/
  - Download page: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

- Twitter US Airline Sentiment Dataset
  - Download page: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

- Sentiment140 Dataset
  - Download page: https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip

Expected columns:

- IMDb: `review`, `sentiment` (values: `positive` / `negative`)
- Twitter US Airline: `text`, `airline_sentiment` (values: `positive` / `neutral` / `negative`)
- Sentiment140: either the default 6-column CSV (`target,id,date,query,user,text`) or an equivalent CSV containing at least `target` and `text`.

Preprocessing in this codebase removes:
- URLs
- `@mentions`
