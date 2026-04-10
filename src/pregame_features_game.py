import pandas as pd

from src.config import (
    PREGAME_TEAM_SNAPSHOT_FILE,
    STARTER_GAME_LOGS_FILE,
    TEAM_BATTING_LOGS_FILE,
    PREGAME_FEATURES_GAME_FILE,
)


def load_inputs():
    """
    Carga las tablas base:
    - pregame_team_snapshot: 1 fila por partido
    - starter_game_logs: historial de abridores con rolling
    - team_batting_logs: historial ofensivo por equipo con rolling
    """
    pregame_df = pd.read_csv(PREGAME_TEAM_SNAPSHOT_FILE)
    starter_df = pd.read_csv(STARTER_GAME_LOGS_FILE)
    batting_df = pd.read_csv(TEAM_BATTING_LOGS_FILE)

    pregame_df["game_date"] = pd.to_datetime(pregame_df["game_date"])
    starter_df["game_date"] = pd.to_datetime(starter_df["game_date"])
    batting_df["game_date"] = pd.to_datetime(batting_df["game_date"])

    return pregame_df, starter_df, batting_df


def prepare_starter_history(starter_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deja solo las columnas necesarias del historial de abridores.
    """
    starter_cols = [
        "gamePk",
        "game_date",
        "pitcher_id",
        "pitcher_name",
        "team_id",
        "opponent_team_id",
        "is_home",
        "starts_count_last_3",
        "innings_pitched_outs_last_3_avg",
        "hits_allowed_last_3_avg",
        "earned_runs_last_3_avg",
        "walks_last_3_avg",
        "strikeouts_last_3_avg",
        "home_runs_allowed_last_3_avg",
        "pitches_thrown_last_3_avg",
        "batters_faced_last_3_avg",
        "runs_allowed_last_3_avg",
        "outs_recorded_last_3_avg",
        "starts_count_last_5",
        "innings_pitched_outs_last_5_avg",
        "hits_allowed_last_5_avg",
        "earned_runs_last_5_avg",
        "walks_last_5_avg",
        "strikeouts_last_5_avg",
        "home_runs_allowed_last_5_avg",
        "pitches_thrown_last_5_avg",
        "batters_faced_last_5_avg",
        "runs_allowed_last_5_avg",
        "outs_recorded_last_5_avg",
    ]

    history_df = starter_df[starter_cols].copy()
    history_df = history_df.sort_values(["pitcher_id", "game_date", "gamePk"]).reset_index(drop=True)

    return history_df

def prepare_batting_history(batting_df: pd.DataFrame) -> pd.DataFrame:
    """
    Deja solo las columnas ofensivas que queremos unir al pregame.
    Esta tabla ya viene con rolling pregame calculado.
    """
    batting_cols = [
        "gamePk",
        "team_id",
        "team_name",
        "games_played_last_3",
        "games_played_last_5",
        "runs_scored_last_3_avg",
        "runs_scored_last_5_avg",
        "hits_last_3_avg",
        "hits_last_5_avg",
        "doubles_last_3_avg",
        "doubles_last_5_avg",
        "triples_last_3_avg",
        "triples_last_5_avg",
        "home_runs_last_3_avg",
        "home_runs_last_5_avg",
        "walks_last_3_avg",
        "walks_last_5_avg",
        "strikeouts_last_3_avg",
        "strikeouts_last_5_avg",
        "singles_last_3_avg",
        "singles_last_5_avg",
        "total_bases_last_3_avg",
        "total_bases_last_5_avg",
        "avg_game_last_3_avg",
        "avg_game_last_5_avg",
        "obp_game_last_3_avg",
        "obp_game_last_5_avg",
        "slg_game_last_3_avg",
        "slg_game_last_5_avg",
        "ops_game_last_3_avg",
        "ops_game_last_5_avg",
    ]

    history_df = batting_df[batting_cols].copy()
    history_df = history_df.sort_values(["team_id", "gamePk"]).reset_index(drop=True)
    return history_df

def get_latest_pitcher_snapshot(
    pitcher_id,
    game_date,
    starter_history_df: pd.DataFrame
):
    """
    Devuelve la última fila disponible de ese pitcher ANTES del partido.

    Regla anti contaminación:
    solo usamos filas con starter_game_date < pregame_game_date
    """
    if pd.isna(pitcher_id):
        return None

    pitcher_history = starter_history_df[
        (starter_history_df["pitcher_id"] == pitcher_id)
        & (starter_history_df["game_date"] < game_date)
    ].copy()

    if pitcher_history.empty:
        return None

    latest_row = pitcher_history.sort_values(["game_date", "gamePk"]).iloc[-1]
    return latest_row


def build_pitcher_snapshot_for_side(
    pregame_df: pd.DataFrame,
    starter_history_df: pd.DataFrame,
    side: str
) -> pd.DataFrame:
    """
    Construye el bloque de features del abridor probable para un lado:
    - away
    - home

    Devuelve:
    1 fila por gamePk con columnas prefijadas.
    """
    pitcher_id_col = f"{side}_probable_pitcher_id"
    pitcher_name_col = f"{side}_probable_pitcher_name"

    rows = []

    for _, row in pregame_df.iterrows():
        game_pk = row["gamePk"]
        game_date = row["game_date"]
        probable_pitcher_id = row[pitcher_id_col]
        probable_pitcher_name = row[pitcher_name_col]

        latest_snapshot = get_latest_pitcher_snapshot(
            pitcher_id=probable_pitcher_id,
            game_date=game_date,
            starter_history_df=starter_history_df
        )

        result_row = {
            "gamePk": game_pk,
            f"{side}_starter_pitcher_id": probable_pitcher_id,
            f"{side}_starter_pitcher_name": probable_pitcher_name,
            f"{side}_starter_context_found_flag": 0,
        }

        if latest_snapshot is not None:
            result_row[f"{side}_starter_context_found_flag"] = 1

            for col in starter_history_df.columns:
                if col in ["gamePk", "game_date", "pitcher_id", "pitcher_name"]:
                    continue

                result_row[f"{side}_starter_{col}"] = latest_snapshot[col]

        rows.append(result_row)

    snapshot_df = pd.DataFrame(rows)
    return snapshot_df


def merge_starter_context(
    pregame_df: pd.DataFrame,
    away_starter_snapshot: pd.DataFrame,
    home_starter_snapshot: pd.DataFrame
) -> pd.DataFrame:
    """
    Une los snapshots de abridores al pregame base.
    La salida debe seguir teniendo 1 fila por partido.
    """
    merged_df = pregame_df.merge(
        away_starter_snapshot,
        on="gamePk",
        how="left",
        validate="one_to_one"
    )

    merged_df = merged_df.merge(
        home_starter_snapshot,
        on="gamePk",
        how="left",
        validate="one_to_one"
    )

    return merged_df

def build_offense_snapshot_for_side(
    pregame_df: pd.DataFrame,
    batting_history_df: pd.DataFrame,
    side: str
) -> pd.DataFrame:
    """
    Construye el bloque ofensivo para away o home.
    Como team_batting_logs ya tiene rolling pregame por gamePk + team_id,
    aquí basta con unir por esas claves.
    """
    team_id_col = f"{side}_team_id"

    merge_df = pregame_df[["gamePk", team_id_col]].copy()

    side_batting_df = batting_history_df.copy()

    merged = merge_df.merge(
        side_batting_df,
        left_on=["gamePk", team_id_col],
        right_on=["gamePk", "team_id"],
        how="left",
        validate="one_to_one"
    )

    rename_map = {
        "team_id": f"{side}_offense_team_id",
        "team_name": f"{side}_offense_team_name",
        "games_played_last_3": f"{side}_offense_games_played_last_3",
        "games_played_last_5": f"{side}_offense_games_played_last_5",
        "runs_scored_last_3_avg": f"{side}_offense_runs_scored_last_3_avg",
        "runs_scored_last_5_avg": f"{side}_offense_runs_scored_last_5_avg",
        "hits_last_3_avg": f"{side}_offense_hits_last_3_avg",
        "hits_last_5_avg": f"{side}_offense_hits_last_5_avg",
        "doubles_last_3_avg": f"{side}_offense_doubles_last_3_avg",
        "doubles_last_5_avg": f"{side}_offense_doubles_last_5_avg",
        "triples_last_3_avg": f"{side}_offense_triples_last_3_avg",
        "triples_last_5_avg": f"{side}_offense_triples_last_5_avg",
        "home_runs_last_3_avg": f"{side}_offense_home_runs_last_3_avg",
        "home_runs_last_5_avg": f"{side}_offense_home_runs_last_5_avg",
        "walks_last_3_avg": f"{side}_offense_walks_last_3_avg",
        "walks_last_5_avg": f"{side}_offense_walks_last_5_avg",
        "strikeouts_last_3_avg": f"{side}_offense_strikeouts_last_3_avg",
        "strikeouts_last_5_avg": f"{side}_offense_strikeouts_last_5_avg",
        "singles_last_3_avg": f"{side}_offense_singles_last_3_avg",
        "singles_last_5_avg": f"{side}_offense_singles_last_5_avg",
        "total_bases_last_3_avg": f"{side}_offense_total_bases_last_3_avg",
        "total_bases_last_5_avg": f"{side}_offense_total_bases_last_5_avg",
        "avg_game_last_3_avg": f"{side}_offense_avg_game_last_3_avg",
        "avg_game_last_5_avg": f"{side}_offense_avg_game_last_5_avg",
        "obp_game_last_3_avg": f"{side}_offense_obp_game_last_3_avg",
        "obp_game_last_5_avg": f"{side}_offense_obp_game_last_5_avg",
        "slg_game_last_3_avg": f"{side}_offense_slg_game_last_3_avg",
        "slg_game_last_5_avg": f"{side}_offense_slg_game_last_5_avg",
        "ops_game_last_3_avg": f"{side}_offense_ops_game_last_3_avg",
        "ops_game_last_5_avg": f"{side}_offense_ops_game_last_5_avg",
    }

    merged = merged.rename(columns=rename_map)

    keep_cols = ["gamePk"] + list(rename_map.values())
    merged = merged[keep_cols].copy()

    merged[f"{side}_offense_context_found_flag"] = (
        merged[f"{side}_offense_team_id"].notna().astype(int)
    )

    return merged


def merge_offense_context(
    pregame_df: pd.DataFrame,
    away_offense_snapshot: pd.DataFrame,
    home_offense_snapshot: pd.DataFrame
) -> pd.DataFrame:
    """
    Une el contexto ofensivo away y home al dataset pregame.
    """
    merged_df = pregame_df.merge(
        away_offense_snapshot,
        on="gamePk",
        how="left",
        validate="one_to_one"
    )

    merged_df = merged_df.merge(
        home_offense_snapshot,
        on="gamePk",
        how="left",
        validate="one_to_one"
    )

    return merged_df

def validate_output(df: pd.DataFrame):
    """
    Validaciones básicas para confirmar que no rompimos el grain.
    """
    print("\n--- VALIDACIONES FINALES ---")
    print("Shape final:", df.shape)
    print("Filas únicas por gamePk:", df["gamePk"].nunique())
    print("Duplicados por gamePk:", df["gamePk"].duplicated().sum())

    print("\nAway context found:")
    print(df["away_starter_context_found_flag"].value_counts(dropna=False))

    print("\nHome context found:")
    print(df["home_starter_context_found_flag"].value_counts(dropna=False))

    print("\nAway usable last 3:")
    print(df["away_starter_has_last_3_data_flag"].value_counts(dropna=False))

    print("\nHome usable last 3:")
    print(df["home_starter_has_last_3_data_flag"].value_counts(dropna=False))

    if "away_offense_context_found_flag" in df.columns:
        print("\nAway offense context found:")
        print(df["away_offense_context_found_flag"].value_counts(dropna=False))

    if "home_offense_context_found_flag" in df.columns:
        print("\nHome offense context found:")
        print(df["home_offense_context_found_flag"].value_counts(dropna=False))

    if "away_offense_has_last_3_data_flag" in df.columns:
        print("\nAway offense usable last 3:")
        print(df["away_offense_has_last_3_data_flag"].value_counts(dropna=False))

    if "home_offense_has_last_3_data_flag" in df.columns:
        print("\nHome offense usable last 3:")
        print(df["home_offense_has_last_3_data_flag"].value_counts(dropna=False))


def main():
    pregame_df, starter_df, batting_df = load_inputs()

    starter_history_df = prepare_starter_history(starter_df)
    batting_history_df = prepare_batting_history(batting_df)

    away_starter_snapshot = build_pitcher_snapshot_for_side(
        pregame_df=pregame_df,
        starter_history_df=starter_history_df,
        side="away"
    )

    home_starter_snapshot = build_pitcher_snapshot_for_side(
        pregame_df=pregame_df,
        starter_history_df=starter_history_df,
        side="home"
    )

    pregame_features_game = merge_starter_context(
        pregame_df=pregame_df,
        away_starter_snapshot=away_starter_snapshot,
        home_starter_snapshot=home_starter_snapshot
    )

    pregame_features_game["away_starter_has_last_3_data_flag"] = (
        pregame_features_game["away_starter_starts_count_last_3"].fillna(0) > 0
    ).astype(int)

    pregame_features_game["away_starter_has_last_5_data_flag"] = (
        pregame_features_game["away_starter_starts_count_last_5"].fillna(0) > 0
    ).astype(int)

    pregame_features_game["home_starter_has_last_3_data_flag"] = (
        pregame_features_game["home_starter_starts_count_last_3"].fillna(0) > 0
    ).astype(int)

    pregame_features_game["home_starter_has_last_5_data_flag"] = (
        pregame_features_game["home_starter_starts_count_last_5"].fillna(0) > 0
    ).astype(int)

    away_offense_snapshot = build_offense_snapshot_for_side(
        pregame_df=pregame_df,
        batting_history_df=batting_history_df,
        side="away"
    )

    home_offense_snapshot = build_offense_snapshot_for_side(
        pregame_df=pregame_df,
        batting_history_df=batting_history_df,
        side="home"
    )

    pregame_features_game = merge_offense_context(
        pregame_df=pregame_features_game,
        away_offense_snapshot=away_offense_snapshot,
        home_offense_snapshot=home_offense_snapshot
    )

    pregame_features_game["away_offense_has_last_3_data_flag"] = (
    pregame_features_game["away_offense_games_played_last_3"].fillna(0) > 0
    ).astype(int)

    pregame_features_game["away_offense_has_last_5_data_flag"] = (
        pregame_features_game["away_offense_games_played_last_5"].fillna(0) > 0
    ).astype(int)

    pregame_features_game["home_offense_has_last_3_data_flag"] = (
        pregame_features_game["home_offense_games_played_last_3"].fillna(0) > 0
    ).astype(int)

    pregame_features_game["home_offense_has_last_5_data_flag"] = (
        pregame_features_game["home_offense_games_played_last_5"].fillna(0) > 0
    ).astype(int)

    validate_output(pregame_features_game)

    pregame_features_game.to_csv(PREGAME_FEATURES_GAME_FILE, index=False)
    print(f"\nArchivo guardado en: {PREGAME_FEATURES_GAME_FILE}")

    print("\nEjemplo columnas nuevas de offense:")
    offense_cols = [c for c in pregame_features_game.columns if "_offense_" in c]
    print(sorted(offense_cols)[:25])

    print("\nVista previa final:")
    preview_cols = [
        "gamePk",
        "game_date",
        "away_team_name",
        "home_team_name",
        "away_offense_context_found_flag",
        "home_offense_context_found_flag",
        "away_offense_games_played_last_3",
        "home_offense_games_played_last_3",
        "away_offense_runs_scored_last_3_avg",
        "home_offense_runs_scored_last_3_avg",
        "away_offense_hits_last_3_avg",
        "home_offense_hits_last_3_avg",
        "away_offense_ops_game_last_3_avg",
        "home_offense_ops_game_last_3_avg",
        "away_offense_runs_scored_last_5_avg",
        "home_offense_runs_scored_last_5_avg",
        "away_offense_ops_game_last_5_avg",
        "home_offense_ops_game_last_5_avg",
        "away_offense_has_last_3_data_flag",
        "home_offense_has_last_3_data_flag",
    ]
    preview_cols = [c for c in preview_cols if c in pregame_features_game.columns]
    print(pregame_features_game[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
