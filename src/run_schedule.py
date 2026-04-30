from pathlib import Path

import pandas as pd

from src.config import START_DATE, END_DATE
from src.pipeline_paths import get_pipeline_paths
from src.pregame_snapshot import build_pregame_team_snapshot, save_pregame_snapshot
from src.schedule import build_schedule_for_range
from src.team_logs import (
    build_team_game_logs,
    validate_team_game_logs,
    add_team_rolling_features,
)


def save_csv(df: pd.DataFrame, output_file) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def build_schedule_pipeline_for_date_range(
    start_date: str,
    end_date: str,
    context_start_date: str | None = None,
    save_output: bool = True,
    verbose: bool = True,
):
    paths = get_pipeline_paths(start_date=start_date, end_date=end_date)
    schedule_start_date = context_start_date or start_date

    df_games = build_schedule_for_range(schedule_start_date, end_date)

    df_team_logs = build_team_game_logs(df_games)
    validate_team_game_logs(df_team_logs, df_games)
    df_team_logs = add_team_rolling_features(df_team_logs, windows=[3, 5, 10])

    df_snapshot_full = build_pregame_team_snapshot(df_games, df_team_logs)

    start_ts = pd.to_datetime(start_date)
    end_ts = pd.to_datetime(end_date)

    df_team_logs["game_date"] = pd.to_datetime(df_team_logs["game_date"], errors="coerce")
    df_snapshot_full["game_date"] = pd.to_datetime(df_snapshot_full["game_date"], errors="coerce")

    in_window_team_logs = (df_team_logs["game_date"] >= start_ts) & (df_team_logs["game_date"] <= end_ts)
    in_window_snapshot = (df_snapshot_full["game_date"] >= start_ts) & (df_snapshot_full["game_date"] <= end_ts)

    df_team_logs = df_team_logs.loc[in_window_team_logs].copy()
    df_snapshot = df_snapshot_full.loc[in_window_snapshot].copy()

    if save_output:
        # games_schedule ya lo guarda schedule.py automáticamente
        team_logs_output_path = save_csv(
            df_team_logs,
            paths["team_game_logs_file"],
        )
        snapshot_output_path = save_pregame_snapshot(
            df_snapshot,
            file_name=paths["pregame_team_snapshot_file"],
        )

        if verbose:
            print(f"team_game_logs guardado en: {team_logs_output_path}")
            print(f"pregame_team_snapshot guardado en: {snapshot_output_path}")

    if verbose:
        print("games_schedule shape:", df_games.shape)
        print("team_game_logs shape:", df_team_logs.shape)
        print("pregame_team_snapshot shape:", df_snapshot.shape)
        print()

        snapshot_preview_cols = [
            "game_date",
            "gamePk",
            "away_team_name",
            "home_team_name",
            "away_runs_scored_last_5_avg",
            "away_runs_allowed_last_5_avg",
            "away_win_pct_last_5",
            "home_runs_scored_last_5_avg",
            "home_runs_allowed_last_5_avg",
            "home_win_pct_last_5",
        ]

        preview_cols = [c for c in snapshot_preview_cols if c in df_snapshot.columns]
        print(df_snapshot[preview_cols].head(20))

    return {
        "games_df": df_games,
        "team_logs_df": df_team_logs,
        "snapshot_df": df_snapshot,
        "paths": paths,
    }


def main():
    build_schedule_pipeline_for_date_range(
        start_date=START_DATE,
        end_date=END_DATE,
        save_output=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
