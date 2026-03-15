# Data folder

Raw dataset files are not included in this repository.
Place these files in `data/raw/`:

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

- IMDb: `review`, `sentiment`
- Twitter US Airline: `text`, `airline_sentiment`
- Sentiment140: 6-column CSV (`target,id,date,query,user,text`) or `target` and `text`.

Preprocessing in this removes:
- URLs
- `@mentions`
