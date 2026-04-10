from pathlib import Path
import pandas as pd

from src.config import GAMES_SCHEDULE_FILE, TEAM_GAME_LOGS_FILE

DEFAULT_OUTPUT_DIR = TEAM_GAME_LOGS_FILE.parent


def load_games_schedule() -> pd.DataFrame:
    return pd.read_csv(GAMES_SCHEDULE_FILE)


def build_team_game_logs(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforma games_schedule (1 fila por partido) en team_game_logs
    (2 filas por partido: una por cada equipo).
    """
    team_rows = []

    for _, row in games_df.iterrows():
        away_row = {
            "game_date": row["game_date"],
            "gamePk": row["gamePk"],
            "game_type": row["game_type"],
            "season": row["season"],
            "game_datetime": row["game_datetime"],

            "team_id": row["away_team_id"],
            "team_name": row["away_team_name"],
            "team_abbreviation": row["away_team_abbreviation"],

            "opponent_team_id": row["home_team_id"],
            "opponent_team_name": row["home_team_name"],
            "opponent_team_abbreviation": row["home_team_abbreviation"],

            "is_home": 0,
            "is_away": 1,

            "venue_id": row["venue_id"],
            "venue_name": row["venue_name"],

            "starter_pitcher_id": row["away_probable_pitcher_id"],
            "starter_pitcher_name": row["away_probable_pitcher_name"],

            "runs_scored": row["away_score_final"],
            "runs_allowed": row["home_score_final"],

            "win_flag": row["away_win_flag"],
            "loss_flag": 1 - row["away_win_flag"] if pd.notna(row["away_win_flag"]) else None,

            "game_status": row["status_abstract_game_state"],
            "is_final": row["is_final"],
            "is_pre_game": row["is_pre_game"],
            "is_in_progress": row["is_in_progress"],
        }

        home_row = {
            "game_date": row["game_date"],
            "gamePk": row["gamePk"],
            "game_type": row["game_type"],
            "season": row["season"],
            "game_datetime": row["game_datetime"],

            "team_id": row["home_team_id"],
            "team_name": row["home_team_name"],
            "team_abbreviation": row["home_team_abbreviation"],

            "opponent_team_id": row["away_team_id"],
            "opponent_team_name": row["away_team_name"],
            "opponent_team_abbreviation": row["away_team_abbreviation"],

            "is_home": 1,
            "is_away": 0,

            "venue_id": row["venue_id"],
            "venue_name": row["venue_name"],

            "starter_pitcher_id": row["home_probable_pitcher_id"],
            "starter_pitcher_name": row["home_probable_pitcher_name"],

            "runs_scored": row["home_score_final"],
            "runs_allowed": row["away_score_final"],

            "win_flag": row["home_win_flag"],
            "loss_flag": 1 - row["home_win_flag"] if pd.notna(row["home_win_flag"]) else None,

            "game_status": row["status_abstract_game_state"],
            "is_final": row["is_final"],
            "is_pre_game": row["is_pre_game"],
            "is_in_progress": row["is_in_progress"],
        }

        team_rows.append(away_row)
        team_rows.append(home_row)

    return pd.DataFrame(team_rows)


def validate_team_game_logs(team_logs_df: pd.DataFrame, games_df: pd.DataFrame) -> None:
    """
    Validaciones básicas:
    - no vacío
    - 2 filas por partido
    - no duplicados por gamePk + team_id
    """
    if team_logs_df.empty:
        raise ValueError("team_game_logs está vacío.")

    expected_rows = len(games_df) * 2
    if len(team_logs_df) != expected_rows:
        raise ValueError(
            f"team_game_logs debería tener {expected_rows} filas y tiene {len(team_logs_df)}."
        )

    duplicated = team_logs_df.duplicated(subset=["gamePk", "team_id"])
    if duplicated.any():
        dupes = team_logs_df.loc[duplicated, ["gamePk", "team_id"]]
        raise ValueError(f"Hay duplicados en gamePk + team_id:\n{dupes}")


def add_team_rolling_features(team_logs_df: pd.DataFrame, windows: list[int] = [3, 5]) -> pd.DataFrame:
    """
    Agrega features rolling por equipo usando solo juegos anteriores.
    Solo considera juegos finales para construir el histórico.
    """
    df = team_logs_df.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["game_datetime"] = pd.to_datetime(df["game_datetime"], utc=True, errors="coerce")

    df = df.sort_values(["team_id", "game_datetime", "gamePk"]).reset_index(drop=True)

    # contador de juegos por equipo
    df["team_game_number"] = df.groupby("team_id").cumcount() + 1

    # diferencial de carreras del juego
    df["run_diff"] = df["runs_scored"] - df["runs_allowed"]

    # histórico solo con juegos finales
    final_mask = df["is_final"] == True

    df["runs_scored_for_rolling"] = df["runs_scored"].where(final_mask)
    df["runs_allowed_for_rolling"] = df["runs_allowed"].where(final_mask)
    df["win_flag_for_rolling"] = df["win_flag"].where(final_mask)
    df["run_diff_for_rolling"] = df["run_diff"].where(final_mask)

    group = df.groupby("team_id", group_keys=False)

    for window in windows:
        df[f"runs_scored_last_{window}_avg"] = group["runs_scored_for_rolling"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[f"runs_allowed_last_{window}_avg"] = group["runs_allowed_for_rolling"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[f"wins_last_{window}"] = group["win_flag_for_rolling"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).sum()
        )

        df[f"games_played_last_{window}"] = group["win_flag_for_rolling"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).count()
        )

        df[f"run_diff_last_{window}_avg"] = group["run_diff_for_rolling"].transform(
            lambda s: s.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[f"win_pct_last_{window}"] = (
            df[f"wins_last_{window}"] / df[f"games_played_last_{window}"]
        )

    return df


def save_team_game_logs(team_logs_df: pd.DataFrame, file_name: str | Path = TEAM_GAME_LOGS_FILE) -> Path:
    output_path = Path(file_name)
    if output_path.parent == Path("."):
        output_path = DEFAULT_OUTPUT_DIR / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    team_logs_df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def main():
    games_df = load_games_schedule()

    print("games_df shape:", games_df.shape)

    team_logs_df = build_team_game_logs(games_df)
    validate_team_game_logs(team_logs_df, games_df)
    team_logs_df = add_team_rolling_features(team_logs_df, windows=[3, 5])

    output_path = save_team_game_logs(team_logs_df, TEAM_GAME_LOGS_FILE)

    print("team_game_logs shape:", team_logs_df.shape)
    print(f"\nArchivo guardado en: {output_path}")


if __name__ == "__main__":
    main()
