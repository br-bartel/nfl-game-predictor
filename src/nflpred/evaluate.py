"""Walk-forward baselines: train on every season before the test season, score it,
then compare against the Vegas closing spread.

Run: python -m nflpred.evaluate
"""
import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nflpred.config import FEATURES, REPORTS

TEST_SEASONS = list(range(2019, 2027))
# Typical sd of NFL margin around the spread; converts a predicted margin to P(home win).
MARGIN_SD = 13.5


def feature_cols(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("diff_form_")] + [
        "rest_diff", "neutral_site", "div_game",
        "home_games_this_season", "away_games_this_season",
        "home_qb_changed", "away_qb_changed", "diff_qb_experience",
    ]


def models():
    return {
        "ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "lgbm": lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.02, num_leaves=8, min_child_samples=40,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
        ),
    }


def walk_forward(games: pl.DataFrame) -> pl.DataFrame:
    played = games.filter(pl.col("result").is_not_null() & pl.col("diff_form_off_epa").is_not_null())
    cols = feature_cols(games)
    out = []
    for season in TEST_SEASONS:
        train = played.filter(pl.col("season") < season)
        test = played.filter(pl.col("season") == season)
        if test.is_empty():
            continue
        preds = test.select("game_id", "season", "week", "result", "spread_line", "home_win")
        for name, model in models().items():
            model.fit(train.select(cols).to_numpy(), train["result"].to_numpy())
            preds = preds.with_columns(pl.Series(f"pred_{name}", model.predict(test.select(cols).to_numpy())))
        out.append(preds)
    return pl.concat(out).with_columns(pred_vegas=pl.col("spread_line"), pred_home_field=pl.lit(1.5))


def score(preds: pl.DataFrame, name: str) -> dict:
    p = preds.filter(pl.col("result") != 0)  # ties are ~0.2% of games
    margin = p[f"pred_{name}"].to_numpy()
    y, win, spread = p["result"].to_numpy(), p["home_win"].to_numpy(), p["spread_line"].to_numpy()
    prob = norm.cdf(margin / MARGIN_SD).clip(1e-4, 1 - 1e-4)
    row = {
        "model": name,
        "games": len(p),
        "mae": float(np.mean(np.abs(margin - y))),
        "brier": float(np.mean((prob - win) ** 2)),
        "log_loss": float(-np.mean(win * np.log(prob) + (1 - win) * np.log(1 - prob))),
        "accuracy": float(np.mean((margin > 0) == (win == 1))),
    }
    if name != "vegas":
        # Against the spread: back home when the model likes them more than the market.
        ats = np.sign(margin - spread) * np.sign(y - spread)
        ats = ats[(ats != 0)]
        row["ats_win_rate"] = float(np.mean(ats > 0)) if len(ats) else None
    return row


def main() -> None:
    games = pl.read_parquet(FEATURES / "games.parquet")
    preds = walk_forward(games)
    names = ["home_field", "vegas", "ridge", "lgbm"]
    overall = pl.DataFrame([score(preds, n) for n in names])
    by_season = pl.DataFrame([
        {"season": s, **score(preds.filter(pl.col("season") == s), n)}
        for s in preds["season"].unique().sort() for n in names
    ])
    REPORTS.mkdir(exist_ok=True)
    preds.write_csv(REPORTS / "walk_forward_predictions.csv")
    by_season.write_csv(REPORTS / "baseline_by_season.csv")
    overall.write_csv(REPORTS / "baseline_overall.csv")
    with pl.Config(tbl_rows=50, tbl_cols=20, float_precision=3):
        print(overall)
        print(by_season.select("season", "model", "games", "mae", "brier", "ats_win_rate"))


if __name__ == "__main__":
    main()
