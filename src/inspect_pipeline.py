import pandas as pd

from src.config import PREGAME_FEATURES_GAME_FILE, START_DATE, END_DATE


INSPECT_DATE = END_DATE


def load_pregame_features() -> pd.DataFrame:
    """
    Carga el dataset final pregame por partido.
    """
    df = pd.read_csv(PREGAME_FEATURES_GAME_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def format_number(value, decimals=2):
    """
    Formatea números para impresión amigable.
    """
    if pd.isna(value):
        return "NA"
    return f"{value:.{decimals}f}"


def print_game_block(row: pd.Series):
    """
    Imprime un partido en formato legible para inspección.
    """
    print("=" * 100)
    print(
        f"{row['away_team_name']} @ {row['home_team_name']} | "
        f"gamePk={row['gamePk']} | date={row['game_date'].date()}"
    )
    print("-" * 100)

    print("Pitchers probables")
    print(
        f"  Away: {row.get('away_probable_pitcher_name', 'NA')} "
        f"| context={row.get('away_starter_context_found_flag', 'NA')} "
        f"| usable_last_3={row.get('away_starter_has_last_3_data_flag', 'NA')}"
    )
    print(
        f"  Home: {row.get('home_probable_pitcher_name', 'NA')} "
        f"| context={row.get('home_starter_context_found_flag', 'NA')} "
        f"| usable_last_3={row.get('home_starter_has_last_3_data_flag', 'NA')}"
    )

    print("\nTeam recent form")
    print(
        f"  Away runs scored last 3: {format_number(row.get('away_runs_scored_last_3_avg'))} | "
        f"runs allowed last 3: {format_number(row.get('away_runs_allowed_last_3_avg'))} | "
        f"win pct last 3: {format_number(row.get('away_win_pct_last_3_avg', row.get('away_win_pct_last_3')))}"
    )
    print(
        f"  Home runs scored last 3: {format_number(row.get('home_runs_scored_last_3_avg'))} | "
        f"runs allowed last 3: {format_number(row.get('home_runs_allowed_last_3_avg'))} | "
        f"win pct last 3: {format_number(row.get('home_win_pct_last_3_avg', row.get('home_win_pct_last_3')))}"
    )

    print("\nOffense context")
    print(
        f"  Away offense usable last 3: {row.get('away_offense_has_last_3_data_flag', 'NA')} | "
        f"games played last 3: {format_number(row.get('away_offense_games_played_last_3'), 0)} | "
        f"OPS last 3: {format_number(row.get('away_offense_ops_game_last_3_avg'), 3)} | "
        f"runs last 3: {format_number(row.get('away_offense_runs_scored_last_3_avg'))}"
    )
    print(
        f"  Home offense usable last 3: {row.get('home_offense_has_last_3_data_flag', 'NA')} | "
        f"games played last 3: {format_number(row.get('home_offense_games_played_last_3'), 0)} | "
        f"OPS last 3: {format_number(row.get('home_offense_ops_game_last_3_avg'), 3)} | "
        f"runs last 3: {format_number(row.get('home_offense_runs_scored_last_3_avg'))}"
    )

    print("\nStarter context")
    print(
        f"  Away starter runs allowed last 3: {format_number(row.get('away_starter_runs_allowed_last_3_avg'))} | "
        f"K last 3: {format_number(row.get('away_starter_strikeouts_last_3_avg'))} | "
        f"walks last 3: {format_number(row.get('away_starter_walks_last_3_avg'))}"
    )
    print(
        f"  Home starter runs allowed last 3: {format_number(row.get('home_starter_runs_allowed_last_3_avg'))} | "
        f"K last 3: {format_number(row.get('home_starter_strikeouts_last_3_avg'))} | "
        f"walks last 3: {format_number(row.get('home_starter_walks_last_3_avg'))}"
    )

    print("\nGame context")
    print(
        f"  Status: {row.get('status_detailed_state', 'NA')} | "
        f"Venue: {row.get('venue_name', 'NA')} | "
        f"Game type: {row.get('game_type', 'NA')} | "
        f"Day/Night: {row.get('day_night', 'NA')}"
    )


def inspect_date(df: pd.DataFrame, target_date: str):
    """
    Filtra e imprime todos los partidos de una fecha dada.
    """
    target_ts = pd.to_datetime(target_date)
    date_df = df[df["game_date"] == target_ts].copy()

    print(f"\nArchivo leído: {PREGAME_FEATURES_GAME_FILE}")
    print(f"Rango configurado actual: {START_DATE} -> {END_DATE}")
    print(f"Fecha inspeccionada: {target_ts.date()}")
    print(f"Partidos encontrados: {len(date_df)}")

    if date_df.empty:
        print("\nNo hay partidos para esa fecha en el dataset actual.")
        return

    date_df = date_df.sort_values(["game_date", "away_team_name", "home_team_name"]).reset_index(drop=True)

    for _, row in date_df.iterrows():
        print_game_block(row)

    print("\n" + "=" * 100)
    print("Resumen rápido de la fecha")
    summary_cols = [
        "gamePk",
        "away_team_name",
        "home_team_name",
        "away_probable_pitcher_name",
        "home_probable_pitcher_name",
        "away_offense_ops_game_last_3_avg",
        "home_offense_ops_game_last_3_avg",
        "away_starter_runs_allowed_last_3_avg",
        "home_starter_runs_allowed_last_3_avg",
    ]
    summary_cols = [c for c in summary_cols if c in date_df.columns]
    print(date_df[summary_cols].to_string(index=False))


def main():
    df = load_pregame_features()
    inspect_date(df, INSPECT_DATE)


if __name__ == "__main__":
    main()
