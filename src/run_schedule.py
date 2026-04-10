from src.schedule import build_schedule_for_range
from src.config import START_DATE, END_DATE
from src.pregame_snapshot import build_pregame_team_snapshot, save_pregame_snapshot
from src.team_logs import (
    build_team_game_logs,
    validate_team_game_logs,
    add_team_rolling_features,
    save_team_game_logs,
)


def main():
    start_date = START_DATE
    end_date = END_DATE

    df_games = build_schedule_for_range(start_date, end_date)

    df_team_logs = build_team_game_logs(df_games)
    validate_team_game_logs(df_team_logs, df_games)

    df_team_logs = add_team_rolling_features(df_team_logs, windows=[3, 5])
    save_team_game_logs(df_team_logs)

    df_snapshot = build_pregame_team_snapshot(df_games, df_team_logs)
    save_pregame_snapshot(df_snapshot)

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

    print(df_snapshot[snapshot_preview_cols].head(20))


if __name__ == "__main__":
    main()
