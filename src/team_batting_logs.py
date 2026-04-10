import pandas as pd
import requests

from src.config import GAMES_SCHEDULE_FILE, TEAM_BATTING_LOGS_FILE

BOX_SCORE_URL = "https://statsapi.mlb.com/api/v1/game/{gamePk}/boxscore"


def load_games_schedule() -> pd.DataFrame:
    """
    Carga la tabla de partidos ya construida.
    """
    games_df = pd.read_csv(GAMES_SCHEDULE_FILE)
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
    """
    df = df.copy()
    df = df.sort_values(["team_id", "game_date", "gamePk"]).reset_index(drop=True)

    rolling_metrics = [
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
        "avg_game",
        "obp_game",
        "slg_game",
        "ops_game",
    ]

    windows = [3, 5]

    for window in windows:
        for col in rolling_metrics:
            new_col = f"{col}_last_{window}_avg"

            df[new_col] = (
                df.groupby("team_id")[col]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

        count_col = f"games_played_last_{window}"
        df[count_col] = (
            df.groupby("team_id")["gamePk"]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).count())
        )

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
        "runs_scored",
        "hits",
        "walks",
        "strikeouts",
        "ops_game",
        "runs_scored_last_3_avg",
        "hits_last_3_avg",
        "walks_last_3_avg",
        "strikeouts_last_3_avg",
        "ops_game_last_3_avg",
        "games_played_last_3",
        "runs_scored_last_5_avg",
        "hits_last_5_avg",
        "ops_game_last_5_avg",
        "games_played_last_5",
    ]

    preview_cols = [c for c in preview_cols if c in df.columns]

    print("\nVista previa:")
    print(df[preview_cols].head(12).to_string(index=False))


def main():
    games_df = load_games_schedule()

    print("games_df shape:", games_df.shape)

    team_batting_logs = build_team_batting_logs(games_df)
    team_batting_logs = add_batting_rolling_features(team_batting_logs)

    validate_output(team_batting_logs)

    team_batting_logs.to_csv(TEAM_BATTING_LOGS_FILE, index=False)
    print(f"\nArchivo guardado en: {TEAM_BATTING_LOGS_FILE}")


if __name__ == "__main__":
    main()
