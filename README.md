# NBA & NCAA Women's Basketball Win Probability Model
A win-probability model built for both the NBA and NCAA Women's Basketball, using Elo-based team ratings, gradient-boosted and logistic regression models, calibration validation, and a benchmark comparison against an independently-built model — with an interactive Streamlit app for exploring results.

## What this project does
- Pulls NBA and NCAA WBB play-by-play data (2023-24 through 2025-26
  seasons) via [`sportsdataverse`](https://github.com/sportsdataverse/sportsdataverse-py)
- Engineers win-probability features, including a custom Elo rating system
  (adapted from [FiveThirtyEight's published methodology](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/))
- Trains and evaluates Logistic Regression and XGBoost models per league,
  with rigorous game-level train/test splitting and calibration checks
- Benchmarks the trained model against `sportsdataverse`'s own bundled
  win-probability model, across a random sample of games
- Ships an interactive Streamlit app for replaying any game's win
  probability curve and reviewing model diagnostics

## Results Overview

| League | Model | AUC | Brier Score |
|---|---|---|---|
| NBA | Logistic Regression | 0.856 | 0.154 |
| NBA | XGBoost | 0.850 | 0.158 |
| NCAA WBB | Logistic Regression | 0.926 | 0.106 |
| NCAA WBB | XGBoost | 0.927 | 0.106 |



## Setup
 
```bash
pip install pandas numpy scikit-learn xgboost matplotlib sportsdataverse streamlit plotly
```

## Usage
 
**1. Run the full pipeline** (pulls data, builds features, trains models,
evaluates, saves a reliability-diagram chart):
 
```bash
python Run_Pipeline.py
```
 
First run will download and cache several seasons of play-by-play data —
this can take a while, especially for NCAA WBB (millions of rows). Cached
data is reused on subsequent runs.
 
**2. Benchmark against sportsdataverse's own model**, across 25 random
games:
 
```bash
python Run_Benchmark.py
```
 
Configurable at the top of the file: `LEAGUE`, `SEASON`, `N_GAMES`,
`RANDOM_SEED`, and `SEED_SOURCE` (`"elo"` or `"model"` — see the write-up
for what each isolates).
 
**3. Generate artifacts & Launch the interactive app** :

```bash
python Export_App_Artifacts.py
```

```bash
streamlit run app/streamlit_app.py
```

## Methodology summary
 
- **Features**: score differential, fraction of game remaining, a
  score-diff × time-pressure interaction term, timeout indicators, and an
  Elo-based team strength differential.
- **Elo system**: home-court advantage adjustment, margin-of-victory
  multiplier, and partial (not full) regression toward the mean between
  seasons — carrying real signal forward rather than resetting team
  strength to a blank slate every year.
- **Models**: Logistic Regression (interpretable baseline) and XGBoost
  (handles nonlinearity natively), evaluated on both discrimination (AUC)
  and calibration (Brier score, reliability diagrams) — since a
  win-probability model's whole value proposition is that its probability
  numbers are actually trustworthy, not just well-ranked.
- **Validation discipline**: train/test splits are grouped by `game_id`
  (not a random row split) to prevent plays from the same game leaking
  across the split.




## Acknowledgments
 
- [`sportsdataverse-py`](https://github.com/sportsdataverse/sportsdataverse-py)
  for play-by-play data access and the benchmark win-probability model.
- Elo methodology adapted from
  [FiveThirtyEight's published NBA Elo approach](https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/).