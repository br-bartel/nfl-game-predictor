"""Build one row per game with pre-game features for both teams.

Every feature uses only games played before kickoff: team-game stats are
aggregated in DuckDB, then shifted one game and exponentially weighted per team
across seasons, so early-season rows still carry (decayed) prior-season form.

Run: python -m nflpred.features
"""
import duckdb
import polars as pl

from nflpred.config import FEATURES, RAW

HALF_LIFE_GAMES = 8

TEAM_GAME_SQL = f"""
WITH plays AS (
    SELECT * FROM read_parquet('{RAW}/pbp/*.parquet')
    WHERE (pass = 1 OR rush = 1) AND epa IS NOT NULL AND special = 0
),
offense AS (
    SELECT game_id, posteam AS team, defteam AS opponent,
        count(*)                                  AS off_plays,
        avg(epa)                                  AS off_epa,
        avg(success)                              AS off_sr,
        avg(epa) FILTER (WHERE pass = 1)          AS off_pass_epa,
        avg(epa) FILTER (WHERE rush = 1)          AS off_rush_epa,
        avg(epa) FILTER (WHERE down IN (1, 2))    AS off_early_epa,
        avg(pass) FILTER (WHERE down IN (1, 2) AND wp BETWEEN 0.2 AND 0.8) AS off_early_pass_rate,
        avg(cpoe)                                 AS off_cpoe,
        sum(interception + fumble_lost)           AS off_turnovers
    FROM plays GROUP BY ALL
)
SELECT o.*,
    d.off_epa       AS def_epa,
    d.off_sr        AS def_sr,
    d.off_pass_epa  AS def_pass_epa,
    d.off_rush_epa  AS def_rush_epa,
    d.off_turnovers AS def_takeaways
FROM offense o
JOIN offense d ON d.game_id = o.game_id AND d.team = o.opponent
"""

# Schedules keep relocated franchises' old codes; play-by-play already uses the
# current ones. Normalizing keeps each franchise's form continuous across moves.
FRANCHISE_CODES = {"SD": "LAC", "STL": "LA", "OAK": "LV"}

QB_HALF_LIFE_GAMES = 10
# Shrink QBs with little history toward a backup-level EPA per dropback.
QB_PRIOR_EPA, QB_PRIOR_GAMES = -0.10, 4

QB_GAME_SQL = f"""
SELECT game_id, passer_player_id AS qb_id, count(*) AS dropbacks, avg(epa) AS qb_epa
FROM read_parquet('{RAW}/pbp/*.parquet')
WHERE qb_dropback = 1 AND epa IS NOT NULL AND passer_player_id IS NOT NULL
GROUP BY ALL
HAVING count(*) >= 10
"""

ROLL_COLS = [
    "off_epa", "off_sr", "off_pass_epa", "off_rush_epa", "off_early_epa", "off_early_pass_rate",
    "off_cpoe", "off_turnovers", "def_epa", "def_sr", "def_pass_epa", "def_rush_epa",
    "def_takeaways", "point_diff",
]


def load_schedules() -> pl.DataFrame:
    return pl.read_parquet(RAW / "schedules.parquet").with_columns(
        pl.col("home_team", "away_team").replace(FRANCHISE_CODES)
    )


def team_games(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    stats = con.sql(TEAM_GAME_SQL).pl()
    sched = load_schedules()
    long = pl.concat([
        sched.select(
            "game_id", "season", "week", "gameday",
            team=pl.col("home_team"), qb_id=pl.col("home_qb_id"),
            point_diff=pl.col("result").cast(pl.Float64),
        ),
        sched.select(
            "game_id", "season", "week", "gameday",
            team=pl.col("away_team"), qb_id=pl.col("away_qb_id"),
            point_diff=-pl.col("result").cast(pl.Float64),
        ),
    ])
    # Left join keeps unplayed games so they receive pre-game features too.
    return long.join(stats, on=["game_id", "team"], how="left").sort("team", "gameday", "game_id")


def rolling_form(tg: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        # ewm_mean leaves nulls where the input is null (unplayed games), so
        # forward-fill the last known form into upcoming weeks.
        pl.col(c).shift(1).ewm_mean(half_life=HALF_LIFE_GAMES, ignore_nulls=True)
        .forward_fill().over("team").alias(f"form_{c}")
        for c in ROLL_COLS
    ]
    played = pl.col("point_diff").is_not_null().cast(pl.Int32)
    return tg.with_columns(
        *exprs,
        games_this_season=(played.cum_sum().over("team", "season") - played),
    )


def qb_form(con: duckdb.DuckDBPyConnection, tg: pl.DataFrame) -> pl.DataFrame:
    """Attach the starting QB's own pre-game form, following him across teams.

    Starters come from the schedule for played games. For upcoming games the
    schedule has no starter yet, so we project the team's most recent starter.
    """
    tg = tg.with_columns(
        qb_id=pl.col("qb_id").forward_fill().over("team"),
        gameday=pl.col("gameday").str.to_date(),
    )
    tg = tg.with_columns(
        qb_changed=(pl.col("qb_id") != pl.col("qb_id").shift(1).over("team"))
        .fill_null(False).cast(pl.Int8)
    )
    dates = tg.select("game_id", "gameday").unique()
    history = (
        con.sql(QB_GAME_SQL).pl()
        .join(dates, on="game_id")
        .sort("qb_id", "gameday")
        .with_columns(
            qb_ewm=pl.col("qb_epa").ewm_mean(half_life=QB_HALF_LIFE_GAMES).over("qb_id"),
            qb_games=pl.int_range(1, pl.len() + 1).over("qb_id"),
        )
        .select("qb_id", "gameday", "qb_ewm", "qb_games")
    )
    # Strictly-before as-of join: the QB's form after his last game prior to this one.
    tg = tg.sort("gameday").join_asof(
        history.sort("gameday"), on="gameday", by="qb_id",
        strategy="backward", allow_exact_matches=False,
    )
    n = pl.col("qb_games").fill_null(0)
    return tg.with_columns(
        form_qb_epa=(pl.col("qb_ewm").fill_null(0) * n + QB_PRIOR_EPA * QB_PRIOR_GAMES)
        / (n + QB_PRIOR_GAMES),
        qb_experience=n,
    ).drop("qb_ewm", "qb_games")


def game_rows(form: pl.DataFrame) -> pl.DataFrame:
    sched = load_schedules()
    feat_cols = [c for c in form.columns if c.startswith("form_")] + [
        "games_this_season", "qb_changed", "qb_experience",
    ]
    side = form.select("game_id", "team", *feat_cols)
    home = side.rename({c: f"home_{c}" for c in feat_cols})
    away = side.rename({c: f"away_{c}" for c in feat_cols})
    games = (
        sched.select(
            "game_id", "season", "game_type", "week", "gameday", "home_team", "away_team",
            "result", "spread_line", "total_line", "home_moneyline", "away_moneyline",
            "home_rest", "away_rest", "div_game", "location", "roof",
        )
        .join(home, left_on=["game_id", "home_team"], right_on=["game_id", "team"])
        .join(away, left_on=["game_id", "away_team"], right_on=["game_id", "team"])
    )
    diffs = [
        (pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"diff_{c}") for c in feat_cols
        if c.startswith("form_") or c == "qb_experience"
    ]
    return games.with_columns(
        *diffs,
        rest_diff=pl.col("home_rest") - pl.col("away_rest"),
        neutral_site=(pl.col("location") == "Neutral").cast(pl.Int8),
        home_win=pl.when(pl.col("result").is_null()).then(None)
        .otherwise((pl.col("result") > 0).cast(pl.Int8)),
    ).sort("gameday", "game_id")


def build() -> pl.DataFrame:
    con = duckdb.connect()
    games = game_rows(qb_form(con, rolling_form(team_games(con))))
    FEATURES.mkdir(parents=True, exist_ok=True)
    games.write_parquet(FEATURES / "games.parquet")
    return games


if __name__ == "__main__":
    g = build()
    print(g.shape)
    print(g.filter(pl.col("season") == 2026).select(
        "game_id", "spread_line", "result", "diff_form_off_epa", "diff_form_qb_epa",
        "home_qb_changed", "away_qb_changed",
    ).tail(8))
