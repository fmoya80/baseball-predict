from pathlib import Path

import pandas as pd
import requests

from src.config import GAMES_SCHEDULE_FILE, TEAM_BATTING_LOGS_FILE
from src.pipeline_paths import get_pipeline_paths

BOX_SCORE_URL = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"


def load_games_schedule(
    games_schedule_file=GAMES_SCHEDULE_FILE,
) -> pd.DataFrame:
    """
    Carga la tabla de partidos ya construida.
    """
    games_df = pd.read_csv(games_schedule_file)
    games_df["game_date"] = pd.to_datetime(games_df["game_date"])
    return games_df


def fetch_boxscore(game_pk: int) -> dict:
    """
    Descarga el boxscore de un partido desde MLB Stats API.
    """
    url = BOX_SCORE_URL.format(gamePk=game_pk)
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def safe_divide(numerator, denominator):
    """
    Evita divisiones por cero.
    """
    if denominator is None or denominator == 0:
        return None
    return numerator / denominator

def safe_round(value, digits=3):
    """
    Redondea si el valor existe.
    """
    if value is None or pd.isna(value):
        return None
    return round(value, digits)


def parse_batting_stats(stats_dict: dict) -> dict:
    """
    Toma el bloque batting del boxscore y devuelve métricas limpias.
    """
    at_bats = stats_dict.get("atBats", 0)
    hits = stats_dict.get("hits", 0)
    doubles = stats_dict.get("doubles", 0)
    triples = stats_dict.get("triples", 0)
    home_runs = stats_dict.get("homeRuns", 0)
    walks = stats_dict.get("baseOnBalls", 0)
    strikeouts = stats_dict.get("strikeOuts", 0)
    hit_by_pitch = stats_dict.get("hitByPitch", 0)
    sac_flies = stats_dict.get("sacFlies", 0)
    runs_scored = stats_dict.get("runs", 0)

    singles = hits - doubles - triples - home_runs
    total_bases = singles + (2 * doubles) + (3 * triples) + (4 * home_runs)

    avg_game = safe_divide(hits, at_bats)

    obp_denominator = at_bats + walks + hit_by_pitch + sac_flies
    obp_numerator = hits + walks + hit_by_pitch
    obp_game = safe_divide(obp_numerator, obp_denominator)

    slg_game = safe_divide(total_bases, at_bats)

    ops_game = None
    if obp_game is not None and slg_game is not None:
        ops_game = obp_game + slg_game

    return {
        "runs_scored": runs_scored,
        "at_bats": at_bats,
        "hits": hits,
        "doubles": doubles,
        "triples": triples,
        "home_runs": home_runs,
        "walks": walks,
        "strikeouts": strikeouts,
        "hit_by_pitch": hit_by_pitch,
        "sac_flies": sac_flies,
        "singles": singles,
        "total_bases": total_bases,
        "avg_game": avg_game,
        "obp_game": obp_game,
        "slg_game": slg_game,
        "ops_game": ops_game,
    }


def build_team_row(
    game_row: pd.Series,
    side: str,
    batting_stats: dict
) -> dict:
    """
    Construye una fila de team_batting_logs para away o home.
    """
    if side == "away":
        team_id = game_row["away_team_id"]
        team_name = game_row["away_team_name"]
        opponent_team_id = game_row["home_team_id"]
        opponent_team_name = game_row["home_team_name"]
        is_home = 0
        is_away = 1
    else:
        team_id = game_row["home_team_id"]
        team_name = game_row["home_team_name"]
        opponent_team_id = game_row["away_team_id"]
        opponent_team_name = game_row["away_team_name"]
        is_home = 1
        is_away = 0

    row = {
        "gamePk": game_row["gamePk"],
        "game_date": game_row["game_date"],
        "team_id": team_id,
        "team_name": team_name,
        "opponent_team_id": opponent_team_id,
        "opponent_team_name": opponent_team_name,
        "is_home": is_home,
        "is_away": is_away,
    }

    row.update(batting_stats)
    return row


def build_team_batting_logs(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Recorre los partidos y arma 2 filas por juego:
    - away
    - home
    """
    rows = []

    for _, game_row in games_df.iterrows():
        game_pk = game_row["gamePk"]

        try:
            boxscore = fetch_boxscore(game_pk)
        except Exception as e:
            print(f"[ERROR] gamePk {game_pk}: {e}")
            continue

        away_batting_raw = (
            boxscore.get("teams", {})
            .get("away", {})
            .get("teamStats", {})
            .get("batting", {})
        )

        home_batting_raw = (
            boxscore.get("teams", {})
            .get("home", {})
            .get("teamStats", {})
            .get("batting", {})
        )

        away_batting_stats = parse_batting_stats(away_batting_raw)
        home_batting_stats = parse_batting_stats(home_batting_raw)

        away_row = build_team_row(
            game_row=game_row,
            side="away",
            batting_stats=away_batting_stats
        )

        home_row = build_team_row(
            game_row=game_row,
            side="home",
            batting_stats=home_batting_stats
        )

        rows.append(away_row)
        rows.append(home_row)

    batting_logs_df = pd.DataFrame(rows)

    batting_logs_df = batting_logs_df.sort_values(
        ["team_id", "game_date", "gamePk"]
    ).reset_index(drop=True)

    return batting_logs_df


def add_batting_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega rolling pregame por equipo.
    Usamos shift(1) para que el partido actual no contamine su propio snapshot.

    Reglas:
    - métricas de conteo: promedio por juego en ventanas L3 y L5
    - métricas de tasa: construidas desde sumas rolling
    - métricas season: construidas desde acumulados season-to-date pregame
    """
    df = df.copy()
    df = df.sort_values(["team_id", "game_date", "gamePk"]).reset_index(drop=True)

    avg_metrics = [
        "runs_scored",
        "at_bats",
        "hits",
        "doubles",
        "triples",
        "home_runs",
        "walks",
        "strikeouts",
        "hit_by_pitch",
        "sac_flies",
        "singles",
        "total_bases",
    ]

    windows = [3, 5, 10]

    for window in windows:
        # 1) métricas promedio por juego
        for col in avg_metrics:
            new_col = f"{col}_last_{window}_avg"

            df[new_col] = (
                df.groupby("team_id")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

        # 2) métricas acumuladas rolling necesarias para tasas
        sum_metrics = [
            "hits",
            "at_bats",
            "walks",
            "hit_by_pitch",
            "sac_flies",
            "total_bases",
        ]

        for col in sum_metrics:
            sum_col = f"{col}_last_{window}_sum"

            df[sum_col] = (
                df.groupby("team_id")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).sum())
            )

        # 3) cantidad de juegos previos usados
        count_col = f"games_played_last_{window}"
        df[count_col] = (
            df.groupby("team_id")["gamePk"]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).count())
        )

        # 4) tasas construidas desde sumas rolling
        avg_col = f"avg_game_last_{window}_avg"
        obp_col = f"obp_game_last_{window}_avg"
        slg_col = f"slg_game_last_{window}_avg"
        ops_col = f"ops_game_last_{window}_avg"
        iso_col = f"iso_game_last_{window}_avg"

        hits_sum_col = f"hits_last_{window}_sum"
        at_bats_sum_col = f"at_bats_last_{window}_sum"
        walks_sum_col = f"walks_last_{window}_sum"
        hbp_sum_col = f"hit_by_pitch_last_{window}_sum"
        sf_sum_col = f"sac_flies_last_{window}_sum"
        tb_sum_col = f"total_bases_last_{window}_sum"

        avg_values = []
        obp_values = []
        slg_values = []
        ops_values = []
        iso_values = []

        for _, row in df.iterrows():
            hits_sum = row[hits_sum_col]
            at_bats_sum = row[at_bats_sum_col]
            walks_sum = row[walks_sum_col]
            hbp_sum = row[hbp_sum_col]
            sf_sum = row[sf_sum_col]
            tb_sum = row[tb_sum_col]

            avg_value = safe_divide(hits_sum, at_bats_sum)

            obp_numerator = hits_sum + walks_sum + hbp_sum
            obp_denominator = at_bats_sum + walks_sum + hbp_sum + sf_sum
            obp_value = safe_divide(obp_numerator, obp_denominator)

            slg_value = safe_divide(tb_sum, at_bats_sum)

            ops_value = None
            if obp_value is not None and slg_value is not None:
                ops_value = obp_value + slg_value

            iso_value = None
            if slg_value is not None and avg_value is not None:
                iso_value = slg_value - avg_value

            avg_values.append(safe_round(avg_value))
            obp_values.append(safe_round(obp_value))
            slg_values.append(safe_round(slg_value))
            ops_values.append(safe_round(ops_value))
            iso_values.append(safe_round(iso_value))

        df[avg_col] = avg_values
        df[obp_col] = obp_values
        df[slg_col] = slg_values
        df[ops_col] = ops_values
        df[iso_col] = iso_values

    # Alias explícitos para L10 solicitados en el master
    df["AVG_last_10"] = df["avg_game_last_10_avg"]
    df["OBP_last_10"] = df["obp_game_last_10_avg"]
    df["SLG_last_10"] = df["slg_game_last_10_avg"]
    df["OPS_last_10"] = df["ops_game_last_10_avg"]
    df["ISO_last_10"] = df["iso_game_last_10_avg"]
    df["hits_last_10"] = df["hits_last_10_avg"]
    df["home_runs_last_10"] = df["home_runs_last_10_avg"]
    df["walks_last_10"] = df["walks_last_10_avg"]
    df["strikeouts_last_10"] = df["strikeouts_last_10_avg"]

    # 5) acumulados season-to-date pregame
    season_sum_metrics = [
        "runs_scored",
        "at_bats",
        "hits",
        "doubles",
        "triples",
        "home_runs",
        "walks",
        "strikeouts",
        "hit_by_pitch",
        "sac_flies",
        "singles",
        "total_bases",
    ]

    for col in season_sum_metrics:
        season_col = f"{col}_season_to_date"

        df[season_col] = (
            df.groupby("team_id")[col]
            .transform(lambda s: s.cumsum().shift(1))
        )

    df["games_played_season_to_date"] = (
        df.groupby("team_id")["gamePk"]
        .transform(lambda s: s.expanding().count().shift(1))
    )

    season_avg_values = []
    season_obp_values = []
    season_slg_values = []
    season_ops_values = []
    season_iso_values = []

    for _, row in df.iterrows():
        hits_sum = row["hits_season_to_date"]
        at_bats_sum = row["at_bats_season_to_date"]
        walks_sum = row["walks_season_to_date"]
        hbp_sum = row["hit_by_pitch_season_to_date"]
        sf_sum = row["sac_flies_season_to_date"]
        tb_sum = row["total_bases_season_to_date"]

        avg_value = safe_divide(hits_sum, at_bats_sum)

        obp_numerator = hits_sum + walks_sum + hbp_sum
        obp_denominator = at_bats_sum + walks_sum + hbp_sum + sf_sum
        obp_value = safe_divide(obp_numerator, obp_denominator)

        slg_value = safe_divide(tb_sum, at_bats_sum)

        ops_value = None
        if obp_value is not None and slg_value is not None:
            ops_value = obp_value + slg_value

        iso_value = None
        if slg_value is not None and avg_value is not None:
            iso_value = slg_value - avg_value

        season_avg_values.append(safe_round(avg_value))
        season_obp_values.append(safe_round(obp_value))
        season_slg_values.append(safe_round(slg_value))
        season_ops_values.append(safe_round(ops_value))
        season_iso_values.append(safe_round(iso_value))

    df["avg_game_season_to_date"] = season_avg_values
    df["obp_game_season_to_date"] = season_obp_values
    df["slg_game_season_to_date"] = season_slg_values
    df["ops_game_season_to_date"] = season_ops_values
    df["iso_game_season_to_date"] = season_iso_values

    return df


def validate_output(df: pd.DataFrame):
    """
    Validaciones simples para confirmar el grain esperado.
    """
    print("\n--- VALIDACIONES TEAM BATTING LOGS ---")
    print("Shape final:", df.shape)
    print("Partidos únicos:", df["gamePk"].nunique())
    print(
        "Filas por partido esperadas aprox:",
        len(df) / df["gamePk"].nunique() if df["gamePk"].nunique() > 0 else None
    )
    print("Duplicados team_id + gamePk:", df.duplicated(subset=["team_id", "gamePk"]).sum())

    print("\nColumnas:")
    print(sorted(df.columns.tolist()))

    preview_cols = [
        "gamePk",
        "game_date",
        "team_name",
        "opponent_team_name",
        "is_home",
        "games_played_season_to_date",
        "runs_scored_season_to_date",
        "hits_season_to_date",
        "at_bats_season_to_date",
        "walks_season_to_date",
        "total_bases_season_to_date",
        "avg_game_season_to_date",
        "obp_game_season_to_date",
        "slg_game_season_to_date",
        "ops_game_season_to_date",
        "iso_game_season_to_date",
        "avg_game_last_3_avg",
        "obp_game_last_3_avg",
        "slg_game_last_3_avg",
        "ops_game_last_3_avg",
        "iso_game_last_3_avg",
        "avg_game_last_5_avg",
        "obp_game_last_5_avg",
        "slg_game_last_5_avg",
        "ops_game_last_5_avg",
        "iso_game_last_5_avg",
    ]

    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nVista previa:")
    print(df[preview_cols].head(12).to_string(index=False))

def save_team_batting_logs(
    df: pd.DataFrame,
    output_file=TEAM_BATTING_LOGS_FILE,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def build_team_batting_logs_file(
    games_schedule_file=GAMES_SCHEDULE_FILE,
    output_file=TEAM_BATTING_LOGS_FILE,
    save_output: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    games_df = load_games_schedule(games_schedule_file=games_schedule_file)

    if verbose:
        print("games_df shape:", games_df.shape)

    team_batting_logs = build_team_batting_logs(games_df)
    team_batting_logs = add_batting_rolling_features(team_batting_logs)

    if verbose:
        validate_output(team_batting_logs)

    if save_output:
        output_path = save_team_batting_logs(
            team_batting_logs,
            output_file=output_file,
        )
        if verbose:
            print(f"\nArchivo guardado en: {output_path}")

    return team_batting_logs


def build_team_batting_logs_file_for_date_range(
    start_date: str,
    end_date: str,
    save_output: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    paths = get_pipeline_paths(start_date=start_date, end_date=end_date)

    return build_team_batting_logs_file(
        games_schedule_file=paths["games_schedule_file"],
        output_file=paths["team_batting_logs_file"],
        save_output=save_output,
        verbose=verbose,
    )


def main():
    build_team_batting_logs_file()


if __name__ == "__main__":
    main()
