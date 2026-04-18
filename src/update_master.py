from pathlib import Path
from datetime import date, timedelta

import pandas as pd

from src.pipeline_paths import (
    PROCESSED_DIR,
    PREGAME_FEATURES_MASTER_FILE,
    INITIAL_HISTORICAL_START_DATE,
    INCREMENTAL_LOOKBACK_DAYS,
    INCREMENTAL_LOOKAHEAD_DAYS,
)
from src.pregame_features_game import build_pregame_features_game_for_date_range
from src.run_schedule import build_schedule_pipeline_for_date_range
from src.starter_logs import build_starter_logs_file_for_date_range
from src.team_batting_logs import build_team_batting_logs_file_for_date_range


DATA_DIR = PROCESSED_DIR
MASTER_FILE = PREGAME_FEATURES_MASTER_FILE
WINDOW_OUTPUT_FILE = DATA_DIR / "pregame_features_window.parquet"


def get_today() -> date:
    return date.today()

def get_initial_historical_window(today: date) -> tuple[str, str]:
    start_date = INITIAL_HISTORICAL_START_DATE
    end_date = today.isoformat()
    return start_date, end_date


def get_incremental_window(today: date) -> tuple[str, str]:
    start_date = today - timedelta(days=INCREMENTAL_LOOKBACK_DAYS)
    end_date = today + timedelta(days=INCREMENTAL_LOOKAHEAD_DAYS)
    return start_date.isoformat(), end_date.isoformat()


def master_exists() -> bool:
    return MASTER_FILE.exists()

def build_required_intermediate_files(start_date: str, end_date: str) -> None:
    """
    Construye archivos intermedios requeridos antes del bloque final.
    """
    print("Construyendo schedule pipeline para la ventana...")

    build_schedule_pipeline_for_date_range(
        start_date=start_date,
        end_date=end_date,
        save_output=True,
        verbose=True,
    )

    print("Construyendo starter_game_logs para la ventana...")

    build_starter_logs_file_for_date_range(
        start_date=start_date,
        end_date=end_date,
        save_output=True,
        verbose=True,
    )

    print("Construyendo team_batting_logs para la ventana...")

    build_team_batting_logs_file_for_date_range(
        start_date=start_date,
        end_date=end_date,
        save_output=True,
        verbose=True,
    )


def build_window_block(start_date: str, end_date: str) -> Path:
    """
    Construye un bloque nuevo de pregame_features para una ventana
    específica y lo guarda en parquet temporal.
    """
    print("Construyendo bloque nuevo desde pregame_features_game...")
    print(f"Rango bloque: {start_date} -> {end_date}")

    build_required_intermediate_files(
        start_date=start_date,
        end_date=end_date,
    )

    df = build_pregame_features_game_for_date_range(
        start_date=start_date,
        end_date=end_date,
        save_output=True,
        verbose=True,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(WINDOW_OUTPUT_FILE, index=False)

    print(f"Bloque temporal guardado en: {WINDOW_OUTPUT_FILE}")
    print(f"Filas bloque temporal: {len(df)}")

    return WINDOW_OUTPUT_FILE

def create_initial_master(window_file: Path) -> Path:
    """
    Crea el master inicial a partir del bloque temporal recién construido.
    """
    df = pd.read_parquet(window_file)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(MASTER_FILE, index=False)

    print(f"Master inicial guardado en: {MASTER_FILE}")
    print(f"Filas master inicial: {len(df)}")

    return MASTER_FILE

def update_existing_master(
    window_file: Path,
    window_start_date: str,
    window_end_date: str,
) -> Path:
    """
    Actualiza el master reemplazando solo la ventana indicada
    por el bloque recién reconstruido.
    """
    master_df = pd.read_parquet(MASTER_FILE)
    window_df = pd.read_parquet(window_file)

    master_df["game_date"] = pd.to_datetime(master_df["game_date"])
    window_df["game_date"] = pd.to_datetime(window_df["game_date"])

    window_start = pd.to_datetime(window_start_date)
    window_end = pd.to_datetime(window_end_date)

    master_outside_window = master_df[
        (master_df["game_date"] < window_start) | (master_df["game_date"] > window_end)
    ].copy()

    updated_master = pd.concat(
        [master_outside_window, window_df],
        ignore_index=True
    )

    updated_master = updated_master.sort_values(["game_date", "gamePk"]).drop_duplicates(
        subset=["gamePk"],
        keep="last"
    ).reset_index(drop=True)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    updated_master.to_parquet(MASTER_FILE, index=False)

    print(f"Master actualizado guardado en: {MASTER_FILE}")
    print(f"Filas master previo: {len(master_df)}")
    print(f"Filas bloque ventana: {len(window_df)}")
    print(f"Filas master fuera ventana: {len(master_outside_window)}")
    print(f"Filas master final: {len(updated_master)}")

    return MASTER_FILE

def print_master_summary(master_file: Path) -> None:
    """
    Imprime un resumen final del master oficial.
    """
    df = pd.read_parquet(master_file)

    if df.empty:
        print("Resumen final master: archivo vacío")
        return

    game_dates = pd.to_datetime(df["game_date"])

    print("-" * 60)
    print("RESUMEN FINAL MASTER")
    print(f"Archivo: {master_file}")
    print(f"Filas: {len(df)}")
    print(f"Partidos únicos: {df['gamePk'].nunique()}")
    print(f"Rango game_date: {game_dates.min().date()} -> {game_dates.max().date()}")
    print("-" * 60)


def run_update_master() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    today = get_today()

    print("=" * 60)
    print("BASEBALL-PREDICT | UPDATE MASTER")
    print("=" * 60)
    print(f"Fecha ejecución: {today.isoformat()}")
    print(f"Archivo master oficial: {MASTER_FILE}")

    if not master_exists():
        print("Estado master: no existe")
        print("Acción: construir master histórico inicial")

        start_date, end_date = get_initial_historical_window(today)

        print(f"Rango inicial histórico: {start_date} -> {end_date}")

        window_file = build_window_block(
            start_date=start_date,
            end_date=end_date,
        )
        master_file = create_initial_master(window_file)

        print_master_summary(master_file)
        print("Resultado: master histórico inicial construido correctamente")
        print("=" * 60)
        return master_file
    start_date, end_date = get_incremental_window(today)

    print("Estado master: existente")
    print("Acción: actualización incremental")
    print(f"Ventana a recalcular: {start_date} -> {end_date}")

    window_file = build_window_block(
        start_date=start_date,
        end_date=end_date,
    )
    master_file = update_existing_master(
        window_file=window_file,
        window_start_date=start_date,
        window_end_date=end_date,
    )

    print_master_summary(master_file)
    print("Resultado: master actualizado correctamente")
    print("=" * 60)
    return master_file


if __name__ == "__main__":
    run_update_master()