from pathlib import Path

DATA_DIR = Path("data/interim")

START_DATE = "2026-03-21"
END_DATE = "2026-04-12"

DATE_RANGE_LABEL = f"{START_DATE}_to_{END_DATE}"

GAMES_SCHEDULE_FILE = DATA_DIR / f"games_schedule_{DATE_RANGE_LABEL}.csv"
TEAM_GAME_LOGS_FILE = DATA_DIR / f"team_game_logs_{DATE_RANGE_LABEL}.csv"
STARTER_GAME_LOGS_FILE = DATA_DIR / f"starter_game_logs_{DATE_RANGE_LABEL}.csv"
TEAM_BATTING_LOGS_FILE = DATA_DIR / f"team_batting_logs_{DATE_RANGE_LABEL}.csv"
PREGAME_TEAM_SNAPSHOT_FILE = DATA_DIR / f"pregame_team_snapshot_{DATE_RANGE_LABEL}.csv"
PREGAME_FEATURES_GAME_FILE = DATA_DIR / f"pregame_features_game_{DATE_RANGE_LABEL}.csv"