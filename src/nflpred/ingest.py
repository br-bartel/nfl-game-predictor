"""Pull raw play-by-play and schedules from nflverse into local parquet files.

Run: python -m nflpred.ingest [--seasons 2015-2026]
Past seasons are written once; the current season is always refreshed.
"""
import argparse
import datetime as dt

import nflreadpy as nfl
import polars as pl

from nflpred.config import FIRST_SEASON, PBP_COLUMNS, RAW


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    # The NFL season starts in September; Jan-Aug games belong to the prior season.
    return today.year if today.month >= 9 else today.year - 1


def parse_seasons(spec: str | None) -> list[int]:
    if not spec:
        return list(range(FIRST_SEASON, current_season() + 1))
    start, _, end = spec.partition("-")
    return list(range(int(start), int(end or start) + 1))


def ingest_pbp(seasons: list[int], refresh: set[int]) -> None:
    out_dir = RAW / "pbp"
    out_dir.mkdir(parents=True, exist_ok=True)
    for season in seasons:
        path = out_dir / f"pbp_{season}.parquet"
        if path.exists() and season not in refresh:
            continue
        df = nfl.load_pbp(season)
        cols = [c for c in PBP_COLUMNS if c in df.columns]
        df.select(cols).write_parquet(path)
        print(f"pbp {season}: {df.height:,} plays -> {path.name}")


def ingest_schedules(seasons: list[int]) -> None:
    df = nfl.load_schedules(seasons)
    path = RAW / "schedules.parquet"
    df.write_parquet(path)
    print(f"schedules: {df.height:,} games -> {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", help="e.g. 2015-2026 or 2026")
    args = parser.parse_args()
    seasons = parse_seasons(args.seasons)
    ingest_pbp(seasons, refresh={current_season()})
    ingest_schedules(seasons)


if __name__ == "__main__":
    main()
