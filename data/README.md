# (raw datasets are not included)

This repository does not redistribute the raw datasets used in the paper.
Please download them from their original sources and place the raw files under:

- `data/raw/IMDB Dataset.csv`
- `data/raw/Twitter US Airline Sentiment Dataset.csv`
- `data/raw/Sentiment140 Dataset.csv`

Expected columns:

- IMDb: `review`, `sentiment` (values: `positive` / `negative`)
- Twitter US Airline: `text`, `airline_sentiment` (values: `positive` / `neutral` / `negative`)
- Sentiment140: either the default 6-column CSV (`target,id,date,query,user,text`) or an equivalent CSV containing at least `target` and `text`.

Preprocessing in this codebase removes:
- URLs
- `@mentions`
