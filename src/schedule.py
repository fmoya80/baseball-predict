from pathlib import Path
import json
from typing import Optional

import pandas as pd
import requests


BASE_URL = "https://statsapi.mlb.com/api/v1/schedule"
RAW_DIR = Path("data/raw")
INTERIM_DIR = Path("data/interim")


def fetch_schedule_raw(
    schedule_date: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """
    Consulta el endpoint schedule de MLB Stats API.

    Puede usarse de dos formas:
    - una fecha puntual: schedule_date="2026-04-07"
    - un rango: start_date="2026-04-01", end_date="2026-04-07"
    """
    params = {
        "sportId": 1,
        "hydrate": "probablePitcher,venue,team",
    }

    if schedule_date:
        params["date"] = schedule_date
    elif start_date and end_date:
        params["startDate"] = start_date
        params["endDate"] = end_date
    else:
        raise ValueError("Debes indicar schedule_date o start_date + end_date.")

    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def save_raw_json(data: dict, file_name: str) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RAW_DIR / file_name

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return output_path


def flatten_schedule_data(raw_data: dict) -> pd.DataFrame:
    rows = []

    for date_block in raw_data.get("dates", []):
        game_date = date_block.get("date")

        for game in date_block.get("games", []):
            teams = game.get("teams", {})
            away = teams.get("away", {})
            home = teams.get("home", {})

            away_team = away.get("team", {})
            home_team = home.get("team", {})

            away_probable_pitcher = away.get("probablePitcher", {})
            home_probable_pitcher = home.get("probablePitcher", {})

            venue = game.get("venue", {})
            status = game.get("status", {})

            away_score = away.get("score")
            home_score = home.get("score")

            abstract_state = status.get("abstractGameState")
            detailed_state = status.get("detailedState")

            is_final = abstract_state == "Final"
            is_pre_game = abstract_state == "Preview"
            is_in_progress = abstract_state == "Live"

            winner_team_id = None
            if is_final and away_score is not None and home_score is not None:
                if away_score > home_score:
                    winner_team_id = away_team.get("id")
                elif home_score > away_score:
                    winner_team_id = home_team.get("id")

            home_win_flag = None
            away_win_flag = None
            if is_final and winner_team_id is not None:
                home_win_flag = int(winner_team_id == home_team.get("id"))
                away_win_flag = int(winner_team_id == away_team.get("id"))

            row = {
                "game_date": game_date,
                "gamePk": game.get("gamePk"),
                "game_type": game.get("gameType"),
                "season": game.get("season"),
                "game_datetime": game.get("gameDate"),

                "status_abstract_game_state": abstract_state,
                "status_abstract_game_code": status.get("abstractGameCode"),
                "status_detailed_state": detailed_state,
                "status_coded_game_state": status.get("codedGameState"),
                "status_status_code": status.get("statusCode"),

                "is_final": is_final,
                "is_pre_game": is_pre_game,
                "is_in_progress": is_in_progress,

                "doubleheader": game.get("doubleHeader"),
                "day_night": game.get("dayNight"),
                "series_game_number": game.get("seriesGameNumber"),
                "scheduled_innings": game.get("scheduledInnings"),

                "away_team_id": away_team.get("id"),
                "away_team_name": away_team.get("name"),
                "away_team_abbreviation": away_team.get("abbreviation"),

                "home_team_id": home_team.get("id"),
                "home_team_name": home_team.get("name"),
                "home_team_abbreviation": home_team.get("abbreviation"),

                "venue_id": venue.get("id"),
                "venue_name": venue.get("name"),

                "away_probable_pitcher_id": away_probable_pitcher.get("id"),
                "away_probable_pitcher_name": away_probable_pitcher.get("fullName"),

                "home_probable_pitcher_id": home_probable_pitcher.get("id"),
                "home_probable_pitcher_name": home_probable_pitcher.get("fullName"),

                "away_score_final": away_score,
                "home_score_final": home_score,
                "winner_team_id": winner_team_id,
                "home_win_flag": home_win_flag,
                "away_win_flag": away_win_flag,
            }

            rows.append(row)

    return pd.DataFrame(rows)

def deduplicate_schedule_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplica games_schedule por gamePk, priorizando:
    Final > Live > Preview
    """
    df = df.copy()

    status_priority_map = {
        "Final": 3,
        "Live": 2,
        "Preview": 1,
    }

    df["status_priority"] = df["status_abstract_game_state"].map(status_priority_map).fillna(0)

    df = df.sort_values(
        ["gamePk", "status_priority", "game_datetime"],
        ascending=[True, False, False]
    )

    df = df.drop_duplicates(subset=["gamePk"], keep="first").reset_index(drop=True)

    df = df.drop(columns=["status_priority"])

    return df


def validate_schedule_df(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("El DataFrame está vacío.")

    if df["gamePk"].isna().any():
        raise ValueError("Hay gamePk nulos.")

    if df["gamePk"].duplicated().any():
        duplicated = df[df["gamePk"].duplicated(keep=False)].sort_values("gamePk")
        raise ValueError(f"Hay gamePk duplicados:\n{duplicated[['gamePk']]}")

    if df["game_date"].isna().any():
        raise ValueError("Hay game_date nulas.")

    invalid_same_team = df[df["away_team_id"] == df["home_team_id"]]
    if not invalid_same_team.empty:
        raise ValueError("Hay partidos donde away_team_id == home_team_id.")


def save_interim_csv(df: pd.DataFrame, file_name: str) -> Path:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    output_path = INTERIM_DIR / file_name
    df.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def build_schedule_for_date(schedule_date: str) -> pd.DataFrame:
    raw_data = fetch_schedule_raw(schedule_date=schedule_date)
    save_raw_json(raw_data, f"schedule_{schedule_date}.json")

    df = flatten_schedule_data(raw_data)
    df = deduplicate_schedule_df(df)
    validate_schedule_df(df)
    save_interim_csv(df, f"games_schedule_{schedule_date}.csv")

    return df


def build_schedule_for_range(start_date: str, end_date: str) -> pd.DataFrame:
    raw_data = fetch_schedule_raw(start_date=start_date, end_date=end_date)
    save_raw_json(raw_data, f"schedule_{start_date}_to_{end_date}.json")

    df = flatten_schedule_data(raw_data)
    df = deduplicate_schedule_df(df)
    validate_schedule_df(df)
    save_interim_csv(df, f"games_schedule_{start_date}_to_{end_date}.csv")

    return df