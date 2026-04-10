from pathlib import Path
import pandas as pd

from src.config import (
    GAMES_SCHEDULE_FILE,
    TEAM_GAME_LOGS_FILE,
    PREGAME_TEAM_SNAPSHOT_FILE,
)

DEFAULT_OUTPUT_DIR = PREGAME_TEAM_SNAPSHOT_FILE.parent


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    games_df = pd.read_csv(GAMES_SCHEDULE_FILE)
    team_logs_df = pd.read_csv(TEAM_GAME_LOGS_FILE)
    return games_df, team_logs_df


def build_pregame_team_snapshot(games_df: pd.DataFrame, team_logs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla pregame con una fila por partido, uniendo
    el contexto rolling del equipo away y del equipo home.
    """
    games = games_df.copy()
    team_logs = team_logs_df.copy()

    key_cols = [
        "gamePk",
        "team_id",
        "team_name",
        "team_abbreviation",
        "runs_scored_last_3_avg",
        "runs_allowed_last_3_avg",
        "run_diff_last_3_avg",
        "wins_last_3",
        "win_pct_last_3",
        "runs_scored_last_5_avg",
        "runs_allowed_last_5_avg",
        "run_diff_last_5_avg",
        "wins_last_5",
        "win_pct_last_5",
        "team_game_number",
    ]

    team_context = team_logs[key_cols].copy()

    away_context = team_context.rename(
        columns={
            "team_id": "away_team_id",
            "team_name": "away_team_name_log",
            "team_abbreviation": "away_team_abbreviation_log",
            "runs_scored_last_3_avg": "away_runs_scored_last_3_avg",
            "runs_allowed_last_3_avg": "away_runs_allowed_last_3_avg",
            "run_diff_last_3_avg": "away_run_diff_last_3_avg",
            "wins_last_3": "away_wins_last_3",
            "win_pct_last_3": "away_win_pct_last_3",
            "runs_scored_last_5_avg": "away_runs_scored_last_5_avg",
            "runs_allowed_last_5_avg": "away_runs_allowed_last_5_avg",
            "run_diff_last_5_avg": "away_run_diff_last_5_avg",
            "wins_last_5": "away_wins_last_5",
            "win_pct_last_5": "away_win_pct_last_5",
            "team_game_number": "away_team_game_number",
        }
    )

    home_context = team_context.rename(
        columns={
            "team_id": "home_team_id",
            "team_name": "home_team_name_log",
            "team_abbreviation": "home_team_abbreviation_log",
            "runs_scored_last_3_avg": "home_runs_scored_last_3_avg",
            "runs_allowed_last_3_avg": "home_runs_allowed_last_3_avg",
            "run_diff_last_3_avg": "home_run_diff_last_3_avg",
            "wins_last_3": "home_wins_last_3",
            "win_pct_last_3": "home_win_pct_last_3",
            "runs_scored_last_5_avg": "home_runs_scored_last_5_avg",
            "runs_allowed_last_5_avg": "home_runs_allowed_last_5_avg",
            "run_diff_last_5_avg": "home_run_diff_last_5_avg",
            "wins_last_5": "home_wins_last_5",
            "win_pct_last_5": "home_win_pct_last_5",
            "team_game_number": "home_team_game_number",
        }
    )

    snapshot = games.merge(
        away_context,
        on=["gamePk", "away_team_id"],
        how="left"
    ).merge(
        home_context,
        on=["gamePk", "home_team_id"],
        how="left"
    )

    return snapshot


def save_pregame_snapshot(df: pd.DataFrame, file_name: str | Path = PREGAME_TEAM_SNAPSHOT_FILE) -> Path:
    output_path = Path(file_name)
    if output_path.parent == Path("."):
        output_path = DEFAULT_OUTPUT_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main():
    games_df, team_logs_df = load_inputs()
    snapshot_df = build_pregame_team_snapshot(games_df, team_logs_df)
    output_path = save_pregame_snapshot(snapshot_df, PREGAME_TEAM_SNAPSHOT_FILE)

    print("games_df shape:", games_df.shape)
    print("team_logs_df shape:", team_logs_df.shape)
    print("pregame_team_snapshot shape:", snapshot_df.shape)
    print(f"\nArchivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
