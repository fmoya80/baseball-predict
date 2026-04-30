from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import GAMES_SCHEDULE_FILE, STARTER_GAME_LOGS_FILE
from src.pipeline_paths import get_pipeline_paths

import pandas as pd
import requests

BASE_URL = "https://statsapi.mlb.com/api/v1"


def fetch_game_boxscore(game_pk: int, timeout: int = 30) -> Dict[str, Any]:
    """
    Descarga el boxscore de un juego MLB usando gamePk.
    """
    url = f"{BASE_URL}/game/{game_pk}/boxscore"
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_many_boxscores(
    game_pks: List[int],
    sleep_seconds: float = 0.1
) -> Dict[int, Dict[str, Any]]:
    """
    Descarga varios boxscores y devuelve:
    {gamePk: boxscore_json}
    """
    results: Dict[int, Dict[str, Any]] = {}

    for i, game_pk in enumerate(game_pks, start=1):
        try:
            if i % 10 == 1 or i == len(game_pks):
                print(f"Descargando boxscore {i}/{len(game_pks)} | gamePk={game_pk}")

            results[game_pk] = fetch_game_boxscore(game_pk)
            time.sleep(sleep_seconds)

        except requests.HTTPError as e:
            print(f"[ERROR] gamePk={game_pk} HTTPError: {e}")
        except requests.RequestException as e:
            print(f"[ERROR] gamePk={game_pk} RequestException: {e}")
        except Exception as e:
            print(f"[ERROR] gamePk={game_pk} Unexpected error: {e}")

    return results


def safe_get(d: dict, *keys, default=None):
    """
    Navegación segura en diccionarios anidados.
    """
    current = d
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current

def resolve_game_status_column(df: pd.DataFrame) -> str:
    candidates = [
        "status_detailed_state",
        "detailed_state",
        "game_status",
        "status",
        "status_abstract_game_state",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "No encontré columna de estado del juego. "
        f"Columnas disponibles: {df.columns.tolist()}"
    )

def select_starter_pitcher(side_data: dict) -> Optional[dict]:
    """
    Selecciona el abridor del equipo de forma más robusta.

    Prioridad:
    1. pitchers del lado cuyo parentTeamId coincida con el team_id
    2. entre ellos, el que tenga gamesStarted == 1
    3. si no existe, el primer pitcher válido del equipo
    4. si no hay match por parentTeamId, fallback al primer pitcher con gamesStarted == 1
    5. último fallback: primer pitcher disponible
    """
    team_id = safe_get(side_data, "team", "id")
    pitchers = side_data.get("pitchers", [])
    players = side_data.get("players", {})

    candidates = []

    for pitcher_id in pitchers:
        player_key = f"ID{pitcher_id}"
        pitcher_data = players.get(player_key, {})
        if not pitcher_data:
            continue

        pitching_stats = safe_get(pitcher_data, "stats", "pitching", default={})
        parent_team_id = pitcher_data.get("parentTeamId")

        candidates.append(
            {
                "pitcher_id": pitcher_id,
                "pitcher_data": pitcher_data,
                "pitching_stats": pitching_stats,
                "parent_team_id": parent_team_id,
                "team_match_flag": 1 if parent_team_id == team_id else 0,
                "games_started": pitching_stats.get("gamesStarted", 0),
            }
        )

    if not candidates:
        return None

    matching_team = [c for c in candidates if c["team_match_flag"] == 1]

    started_matching_team = [c for c in matching_team if c["games_started"] == 1]
    if started_matching_team:
        selected = started_matching_team[0]
        selected["starter_selection_rule"] = "team_match_and_games_started"
        return selected

    if matching_team:
        selected = matching_team[0]
        selected["starter_selection_rule"] = "team_match_fallback_first"
        return selected

    started_any = [c for c in candidates if c["games_started"] == 1]
    if started_any:
        selected = started_any[0]
        selected["starter_selection_rule"] = "games_started_fallback_any_team"
        return selected

    selected = candidates[0]
    selected["starter_selection_rule"] = "first_pitcher_fallback"
    return selected


def innings_pitched_to_outs(ip_value) -> Optional[int]:
    """
    Convierte innings pitched estilo béisbol a outs.
    Ejemplos:
    - '5.0' -> 15
    - '4.1' -> 13
    - '4.2' -> 14
    """
    if pd.isna(ip_value):
        return None

    ip_str = str(ip_value).strip()

    if ip_str == "":
        return None

    if "." not in ip_str:
        try:
            return int(ip_str) * 3
        except ValueError:
            return None

    whole, frac = ip_str.split(".", 1)

    try:
        whole_innings = int(whole)
        partial_outs = int(frac)
    except ValueError:
        return None

    if partial_outs not in (0, 1, 2):
        return None

    return whole_innings * 3 + partial_outs


def add_starter_rolling_features(starter_game_logs: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega rolling features por pitcher usando solo aperturas anteriores.
    """
    if starter_game_logs.empty:
        return starter_game_logs.copy()

    df = starter_game_logs.copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    df["season"] = pd.to_numeric(df["season"], errors="coerce")
    df["innings_pitched_outs"] = df["innings_pitched"].apply(innings_pitched_to_outs)

    numeric_cols = [
        "innings_pitched_outs",
        "hits_allowed",
        "earned_runs",
        "walks",
        "strikeouts",
        "home_runs_allowed",
        "pitches_thrown",
        "batters_faced",
        "runs_allowed",
        "outs_recorded",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["pitcher_id", "game_date", "gamePk"]).reset_index(drop=True)

    windows = [3, 5]
    base_metrics = [
        "innings_pitched_outs",
        "hits_allowed",
        "earned_runs",
        "walks",
        "strikeouts",
        "home_runs_allowed",
        "pitches_thrown",
        "batters_faced",
        "runs_allowed",
        "outs_recorded",
    ]

    grouped = df.groupby("pitcher_id", group_keys=False)

    for window in windows:
        for metric in base_metrics:
            df[f"{metric}_last_{window}_avg"] = (
                grouped[metric]
                .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).mean())
            )

        df[f"starts_count_last_{window}"] = (
            grouped["gamePk"]
            .transform(lambda s: s.shift(1).rolling(window=window, min_periods=1).count())
        )

    season_grouped = df.groupby(["pitcher_id", "season"], group_keys=False)

    season_sum_metrics = [
        "outs_recorded",
        "hits_allowed",
        "earned_runs",
        "walks",
        "strikeouts",
        "home_runs_allowed",
    ]

    season_rename_map = {
        "outs_recorded": "starter_season_outs",
        "hits_allowed": "starter_season_hits_allowed",
        "earned_runs": "starter_season_earned_runs",
        "walks": "starter_season_walks",
        "strikeouts": "starter_season_strikeouts",
        "home_runs_allowed": "starter_season_home_runs_allowed",
    }

    df["starter_season_starts"] = (
        season_grouped["gamePk"].transform(lambda s: s.expanding().count().shift(1))
    )

    for col in season_sum_metrics:
        new_col = season_rename_map[col]
        df[new_col] = season_grouped[col].transform(lambda s: s.cumsum().shift(1))

    df["starter_season_innings"] = (df["starter_season_outs"] / 3).round(3)

    df["starter_season_era"] = pd.NA
    valid_outs_mask = df["starter_season_outs"].notna() & (df["starter_season_outs"] > 0)
    df.loc[valid_outs_mask, "starter_season_era"] = (
        27 * df.loc[valid_outs_mask, "starter_season_earned_runs"] / df.loc[valid_outs_mask, "starter_season_outs"]
    ).round(3)

    return df

def extract_starter_row_from_side(
    game_pk: int,
    game_date: str,
    season: int,
    game_status: str,
    game_is_final: int,
    side_data: dict,
    opponent_data: dict,
    is_home: bool,
) -> Optional[dict]:
    """
    Extrae 1 fila del abridor de un lado del partido.
    """
    team_id = safe_get(side_data, "team", "id")
    team_name = safe_get(side_data, "team", "name")
    opponent_team_id = safe_get(opponent_data, "team", "id")
    opponent_team_name = safe_get(opponent_data, "team", "name")

    selected = select_starter_pitcher(side_data)
    if selected is None:
        return None

    pitcher_id = selected["pitcher_id"]
    pitcher_data = selected["pitcher_data"]
    pitching_stats = selected["pitching_stats"]

    person = pitcher_data.get("person", {})
    parent_team_id = selected["parent_team_id"]
    team_match_flag = selected["team_match_flag"]
    games_started = selected["games_started"]
    starter_selection_rule = selected["starter_selection_rule"]

    if not pitching_stats:
        return None

    starter_confirmed_flag = 1 if games_started == 1 else 0

    row = {
        "gamePk": game_pk,
        "game_date": game_date,
        "season": season,
        "game_status": game_status,
        "is_final": int(game_is_final),
        "team_id": team_id,
        "team_name": team_name,
        "opponent_team_id": opponent_team_id,
        "opponent_team_name": opponent_team_name,
        "is_home": int(is_home),
        "pitcher_id": pitcher_id,
        "pitcher_name": person.get("fullName"),
        "pitcher_parent_team_id": parent_team_id,
        "pitcher_team_match_flag": team_match_flag,
        "starter_selection_rule": starter_selection_rule,
        "starter_confirmed_flag": starter_confirmed_flag,
        "games_started_in_boxscore": games_started,
        "innings_pitched": pitching_stats.get("inningsPitched"),
        "hits_allowed": pitching_stats.get("hits"),
        "earned_runs": pitching_stats.get("earnedRuns"),
        "walks": pitching_stats.get("baseOnBalls"),
        "strikeouts": pitching_stats.get("strikeOuts"),
        "home_runs_allowed": pitching_stats.get("homeRuns"),
        "pitches_thrown": pitching_stats.get("pitchesThrown"),
        "batters_faced": pitching_stats.get("battersFaced"),
        "runs_allowed": pitching_stats.get("runs"),
        "outs_recorded": pitching_stats.get("outs"),
        "decision_wins": pitching_stats.get("wins"),
        "decision_losses": pitching_stats.get("losses"),
        "summary": pitching_stats.get("summary"),
        "source": "mlb_stats_api_boxscore",
    }

    return row


def extract_starter_rows_from_boxscore(
    game_pk: int,
    boxscore_json: dict,
    game_meta_row: pd.Series,
    status_col: str,
) -> List[dict]:
    """
    Devuelve hasta 2 filas por partido:
    - una para away starter
    - una para home starter
    """
    rows: List[dict] = []

    teams = boxscore_json.get("teams", {})
    away = teams.get("away", {})
    home = teams.get("home", {})

    game_date = game_meta_row["game_date"]
    season = int(game_meta_row["season"])
    game_status = game_meta_row[status_col]
    game_is_final = int(game_meta_row["is_final"]) if "is_final" in game_meta_row else int(game_status == "Final")

    away_row = extract_starter_row_from_side(
        game_pk=game_pk,
        game_date=game_date,
        season=season,
        game_status=game_status,
        game_is_final=game_is_final,
        side_data=away,
        opponent_data=home,
        is_home=False,
    )
    if away_row is not None:
        rows.append(away_row)

    home_row = extract_starter_row_from_side(
        game_pk=game_pk,
        game_date=game_date,
        season=season,
        game_status=game_status,
        game_is_final=game_is_final,
        side_data=home,
        opponent_data=away,
        is_home=True,
    )
    if home_row is not None:
        rows.append(home_row)

    return rows


def build_starter_game_logs(
    games_schedule: pd.DataFrame,
    sleep_seconds: float = 0.1,
    limit_games: Optional[int] = None,
) -> pd.DataFrame:
    """
    Construye starter_game_logs a partir de games_schedule.
    """
    required_cols = ["gamePk", "game_date", "season"]
    missing_cols = [col for col in required_cols if col not in games_schedule.columns]
    if missing_cols:
        raise ValueError(f"Faltan columnas en games_schedule: {missing_cols}")

    status_col = resolve_game_status_column(games_schedule)
    print(f"Usando columna de estado: {status_col}")

    print("\nColumnas disponibles en games_schedule:")
    print(games_schedule.columns.tolist())

    work_df = games_schedule.copy()

    # Por ahora nos enfocamos en juegos ya presentes en el CSV.
    # Más adelante podemos filtrar solo Final si quieres.
    work_df["gamePk"] = work_df["gamePk"].astype(int)
    work_df = work_df.drop_duplicates(subset=["gamePk"]).reset_index(drop=True)

    if limit_games is not None:
        work_df = work_df.head(limit_games).copy()

    game_pks = work_df["gamePk"].tolist()
    boxscores = fetch_many_boxscores(game_pks, sleep_seconds=sleep_seconds)

    all_rows: List[dict] = []

    for _, game_row in work_df.iterrows():
        game_pk = int(game_row["gamePk"])
        boxscore_json = boxscores.get(game_pk)

        if boxscore_json is None:
            continue

        rows = extract_starter_rows_from_boxscore(
        game_pk=game_pk,
        boxscore_json=boxscore_json,
        game_meta_row=game_row,
        status_col=status_col,
    )
        all_rows.extend(rows)

    starter_game_logs = pd.DataFrame(all_rows)

    if starter_game_logs.empty:
        return starter_game_logs

    starter_game_logs = starter_game_logs.sort_values(
        ["pitcher_id", "game_date", "gamePk"]
    ).reset_index(drop=True)

    return starter_game_logs

def save_starter_game_logs(
    df: pd.DataFrame,
    output_file=STARTER_GAME_LOGS_FILE,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


def build_starter_logs_file(
    games_schedule_file=GAMES_SCHEDULE_FILE,
    output_file=STARTER_GAME_LOGS_FILE,
    sleep_seconds: float = 0.05,
    limit_games: Optional[int] = None,
    save_output: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    input_path = Path(games_schedule_file)

    if verbose:
        print(f"Leyendo games_schedule desde: {input_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {input_path}")

    games_schedule = pd.read_csv(input_path)

    starter_game_logs = build_starter_game_logs(
        games_schedule=games_schedule,
        sleep_seconds=sleep_seconds,
        limit_games=limit_games,
    )

    starter_game_logs = add_starter_rolling_features(starter_game_logs)

    if save_output:
        output_path = save_starter_game_logs(
            starter_game_logs,
            output_file=output_file,
        )
        if verbose:
            print("\nstarter_game_logs shape:", starter_game_logs.shape)
            print(f"\nArchivo guardado en: {output_path}")

    return starter_game_logs


def build_starter_logs_file_for_date_range(
    start_date: str,
    end_date: str,
    sleep_seconds: float = 0.05,
    limit_games: Optional[int] = None,
    save_output: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    paths = get_pipeline_paths(start_date=start_date, end_date=end_date)

    return build_starter_logs_file(
        games_schedule_file=paths["games_schedule_file"],
        output_file=paths["starter_game_logs_file"],
        sleep_seconds=sleep_seconds,
        limit_games=limit_games,
        save_output=save_output,
        verbose=verbose,
    )


def main():
    build_starter_logs_file()


if __name__ == "__main__":
    main()
