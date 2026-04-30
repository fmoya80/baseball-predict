import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]

DEFAULT_DATA_ROOT = BASE_DIR / "data"
DATA_ROOT = Path(os.getenv("BASEBALL_PREDICT_DATA_DIR", DEFAULT_DATA_ROOT))

INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"

# Output oficial final del pipeline incremental
PREGAME_FEATURES_MASTER_FILE = PROCESSED_DIR / "pregame_features_master.parquet"

# Configuración oficial del runner incremental
INITIAL_HISTORICAL_START_DATE = "2026-03-24"
INCREMENTAL_LOOKBACK_DAYS = 1
INCREMENTAL_LOOKAHEAD_DAYS = 5


def get_date_range_label(start_date: str, end_date: str) -> str:
    return f"{start_date}_to_{end_date}"


def get_pipeline_paths(start_date: str, end_date: str) -> dict:
    """
    Devuelve todas las rutas intermedias del pipeline para un rango de fechas dado.
    """
    date_range_label = get_date_range_label(start_date, end_date)

    return {
        "data_dir": INTERIM_DIR,
        "date_range_label": date_range_label,
        "games_schedule_file": INTERIM_DIR / f"games_schedule_{date_range_label}.csv",
        "team_game_logs_file": INTERIM_DIR / f"team_game_logs_{date_range_label}.csv",
        "starter_game_logs_file": INTERIM_DIR / f"starter_game_logs_{date_range_label}.csv",
        "team_batting_logs_file": INTERIM_DIR / f"team_batting_logs_{date_range_label}.csv",
        "pregame_team_snapshot_file": INTERIM_DIR / f"pregame_team_snapshot_{date_range_label}.csv",
        "pregame_features_game_file": INTERIM_DIR / f"pregame_features_game_{date_range_label}.csv",
    }
