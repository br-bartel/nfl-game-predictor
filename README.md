# NFL Weekly Game Predictor

Predicts the home-team point margin and win probability for every NFL game, using only
information available before kickoff, and scores the predictions against the Vegas closing
spread. The first project in a portfolio series covering classical ML, deep learning, graph ML, and LLMs.

## Status

| Stage | State |
|---|---|
| Ingest (nflreadpy → parquet) | done: 2015–2026, ~545k plays, 3,300 games |
| Features (DuckDB + Polars) | done: team form as exponentially weighted EPA and success rate, with a half-life of 8 games |
| QB features | done: the starter's own form (followed across teams and shrunk toward backup level), a QB-change flag, and experience |
| Walk-forward baselines | done: ridge and LightGBM against Vegas and a home-field-only baseline |
| Tuning, calibration, SHAP | next |
| MLflow, FastAPI, weekly job, monitoring | after that |

## First results (walk-forward, 2019 through 2026 Week 6, 2,032 games)

Each test season is predicted by a model trained only on the seasons before it.

| Model | MAE (pts) | Brier | Log loss | Straight-up accuracy | Against the spread |
|---|---|---|---|---|---|
| Home field only (+1.5) | 11.09 | 0.249 | 0.691 | 53.6% | 51.3% |
| **Vegas closing spread** | **9.84** | **0.212** | **0.611** | **66.2%** | n/a |
| Ridge (team form only) | 10.24 | 0.225 | 0.640 | 63.1% | 51.0% |
| LightGBM (team form only) | 10.38 | 0.227 | 0.646 | 62.9% | 49.9% |
| Ridge (+ QB features) | 10.17 | 0.221 | 0.632 | 64.8% | 49.7% |
| LightGBM (+ QB features) | 10.30 | 0.223 | 0.637 | 64.1% | 49.7% |

The models recover most of the gap between the home-field baseline and Vegas, but the market
still wins, which is the expected starting point. Adding QB features improved every accuracy
metric, but it did not produce an edge against the spread, which suggests the market already prices
in QB changes. Break-even against the spread at standard
-110 odds is 52.4%. Results by season are in `reports/baseline_by_season.csv`.

## Layout

```
src/nflpred/
  config.py     paths, seasons, play-by-play columns kept
  ingest.py     python -m nflpred.ingest   (past seasons cached, current season refreshed)
  features.py   python -m nflpred.features (one row per game, pre-game features only)
  evaluate.py   python -m nflpred.evaluate (walk-forward by season, writes reports/)
data/           raw/ and features/ parquet (gitignored, rebuilt by ingest)
reports/        evaluation outputs
```

## Leakage guards

- Starting QBs for past games come from the schedule. That uses known pre-game information, since starters are almost always announced before kickoff.

- Form features are shifted one game per team before smoothing, so a game never sees its own result.
- Splits are by season (train < test), never random.
- Relocated franchises (SD→LAC, STL→LA, OAK→LV) are mapped so their form carries across the move.

## Next steps

1. Use the injury report (`load_injuries`) to project upcoming starters. Right now, upcoming games assume last week's starter.
2. Make EPA opponent-adjusted, and tune the half-life with time-series CV inside each training window.
3. Calibrate the win probability directly (logistic or isotonic) instead of using the fixed normal-sd conversion.
4. Track experiments in MLflow, serve with FastAPI, and run a weekly scheduled job that publishes predictions before Thursday kickoff.
