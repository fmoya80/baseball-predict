from pathlib import Path
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        print(f"[WARN] No existe: {path}")
        return pd.DataFrame()
    return pd.read_csv(path)


def print_section(title: str):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def preview_df(name: str, df: pd.DataFrame, n: int = 5):
    print_section(name)
    if df.empty:
        print("DataFrame vacío o archivo no encontrado.")
        return

    print("shape:", df.shape)
    print("columns:")
    print(df.columns.tolist())
    print("\nhead:")
    print(df.head(n).to_string(index=False))


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    interim_path = project_root / "data" / "interim"

    games_schedule_path = interim_path / "games_schedule_2026-03-27_to_2026-04-07.csv"
    team_game_logs_path = interim_path / "team_game_logs_2026-03-27_to_2026-04-07.csv"
    pregame_team_snapshot_path = interim_path / "pregame_team_snapshot_2026-03-27_to_2026-04-07.csv"
    starter_game_logs_path = interim_path / "starter_game_logs_2026-03-27_to_2026-04-07.csv"

    games_schedule = load_csv(games_schedule_path)
    team_game_logs = load_csv(team_game_logs_path)
    pregame_team_snapshot = load_csv(pregame_team_snapshot_path)
    starter_game_logs = load_csv(starter_game_logs_path)

    preview_df("games_schedule", games_schedule)
    preview_df("team_game_logs", team_game_logs)
    preview_df("pregame_team_snapshot", pregame_team_snapshot)
    preview_df("starter_game_logs", starter_game_logs)