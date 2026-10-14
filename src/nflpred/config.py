from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
FEATURES = DATA / "features"
REPORTS = ROOT / "reports"

FIRST_SEASON = 2015

# Play-by-play columns we keep. The full table is ~370 columns; this keeps the
# raw store small while covering every feature we build.
PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "home_team", "away_team",
    "play_type", "down", "ydstogo", "yardline_100", "qtr", "game_seconds_remaining",
    "score_differential", "wp", "vegas_wp",
    "epa", "success", "pass", "rush", "qb_dropback", "sack", "interception", "fumble_lost",
    "cpoe", "air_yards", "yards_gained", "penalty", "special",
    "passer_player_id", "passer_player_name",
]
