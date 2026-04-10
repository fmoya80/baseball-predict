from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Baseball Predict",
    layout="wide",
)


DATASET_PATTERNS = (
    "pregame_features_game*.parquet",
    "pregame_features_game*.csv",
)

SEARCH_DIRS = (
    Path("data/processed"),
    Path("data/interim"),
    Path("data"),
    Path("."),
)

DATE_COLUMN_CANDIDATES = [
    "game_date",
    "gameDate",
    "officialDate",
    "date",
    "fecha",
]

MLB_BLUE = "#0C2340"
MLB_RED = "#C8102E"
MLB_GREEN = "#1F8A4C"
MLB_BORDER = "#D9E2EC"
MLB_BG = "#F4F7FB"
MLB_TEXT = "#102A43"

GAME_TABLE_FIELDS = [
    ("gamePk", ["gamePk"], "integer"),
    ("away_team", ["away_team_name", "away_name", "away_team", "team_away_name"], "text"),
    ("home_team", ["home_team_name", "home_name", "home_team", "team_home_name"], "text"),
    (
        "away_pitcher",
        ["away_probable_pitcher_name", "away_starter_pitcher_name", "away_pitcher_name"],
        "text",
    ),
    (
        "home_pitcher",
        ["home_probable_pitcher_name", "home_starter_pitcher_name", "home_pitcher_name"],
        "text",
    ),
    (
        "status",
        [
            "status_detailed_state",
            "status_abstract_game_state",
            "detailedState",
            "game_status",
            "status",
        ],
        "text",
    ),
]

FLAG_FIELDS = [
    ("Starter", "away_starter_has_last_3_data_flag", "home_starter_has_last_3_data_flag"),
    ("Starter", "away_starter_has_last_5_data_flag", "home_starter_has_last_5_data_flag"),
    ("Offense", "away_offense_context_found_flag", "home_offense_context_found_flag"),
    ("Offense", "away_offense_has_last_3_data_flag", "home_offense_has_last_3_data_flag"),
    ("Offense", "away_offense_has_last_5_data_flag", "home_offense_has_last_5_data_flag"),
]

MATCHUP_BASE_FIELDS = {
    "away_team": ["away_team_name", "away_name", "away_team", "team_away_name"],
    "home_team": ["home_team_name", "home_name", "home_team", "team_home_name"],
    "away_pitcher": ["away_probable_pitcher_name", "away_starter_pitcher_name", "away_pitcher_name"],
    "home_pitcher": ["home_probable_pitcher_name", "home_starter_pitcher_name", "home_pitcher_name"],
    "status": [
        "status_detailed_state",
        "status_abstract_game_state",
        "detailedState",
        "game_status",
        "status",
    ],
}

CONTEXT_METRICS = [
    {
        "label": "Equipo",
        "away_candidates": ["away_team_name", "away_name", "away_team", "team_away_name"],
        "home_candidates": ["home_team_name", "home_name", "home_team", "team_home_name"],
        "value_type": "text",
    },
    {
        "label": "Abridor probable",
        "away_candidates": ["away_probable_pitcher_name", "away_starter_pitcher_name", "away_pitcher_name"],
        "home_candidates": ["home_probable_pitcher_name", "home_starter_pitcher_name", "home_pitcher_name"],
        "value_type": "text",
    },
    {
        "label": "Starter con historial L3",
        "away_candidates": ["away_starter_has_last_3_data_flag"],
        "home_candidates": ["home_starter_has_last_3_data_flag"],
        "value_type": "flag",
    },
    {
        "label": "Starter con historial L5",
        "away_candidates": ["away_starter_has_last_5_data_flag"],
        "home_candidates": ["home_starter_has_last_5_data_flag"],
        "value_type": "flag",
    },
    {
        "label": "Contexto ofensivo disponible",
        "away_candidates": ["away_offense_context_found_flag"],
        "home_candidates": ["home_offense_context_found_flag"],
        "value_type": "flag",
    },
    {
        "label": "Ofensiva con historial L3",
        "away_candidates": ["away_offense_has_last_3_data_flag"],
        "home_candidates": ["home_offense_has_last_3_data_flag"],
        "value_type": "flag",
    },
    {
        "label": "Ofensiva con historial L5",
        "away_candidates": ["away_offense_has_last_5_data_flag"],
        "home_candidates": ["home_offense_has_last_5_data_flag"],
        "value_type": "flag",
    },
]

OFFENSE_METRICS_BY_WINDOW = {
    "Temporada": {
        "core": [
            {
                "label": "OPS temporada",
                "away_candidates": ["away_offense_ops_game_season_to_date"],
                "home_candidates": ["home_offense_ops_game_season_to_date"],
                "value_type": "float",
            },
            {
                "label": "ISO temporada",
                "away_candidates": ["away_offense_iso_game_season_to_date"],
                "home_candidates": ["home_offense_iso_game_season_to_date"],
                "value_type": "float",
            },
        ],
        "detail": [
            {
                "label": "Juegos temporada",
                "away_candidates": ["away_offense_games_played_season_to_date"],
                "home_candidates": ["home_offense_games_played_season_to_date"],
                "value_type": "integer",
            },
            {
                "label": "AVG temporada",
                "away_candidates": ["away_offense_avg_game_season_to_date"],
                "home_candidates": ["home_offense_avg_game_season_to_date"],
                "value_type": "float",
            },
            {
                "label": "OBP temporada",
                "away_candidates": ["away_offense_obp_game_season_to_date"],
                "home_candidates": ["home_offense_obp_game_season_to_date"],
                "value_type": "float",
            },
            {
                "label": "SLG temporada",
                "away_candidates": ["away_offense_slg_game_season_to_date"],
                "home_candidates": ["home_offense_slg_game_season_to_date"],
                "value_type": "float",
            },
            {
                "label": "Carreras acumuladas",
                "away_candidates": ["away_offense_runs_scored_season_to_date"],
                "home_candidates": ["home_offense_runs_scored_season_to_date"],
                "value_type": "integer",
            },
            {
                "label": "Hits acumulados",
                "away_candidates": ["away_offense_hits_season_to_date"],
                "home_candidates": ["home_offense_hits_season_to_date"],
                "value_type": "integer",
            },
            {
                "label": "Turnos acumulados",
                "away_candidates": ["away_offense_at_bats_season_to_date"],
                "home_candidates": ["home_offense_at_bats_season_to_date"],
                "value_type": "integer",
            },
            {
                "label": "BB acumuladas",
                "away_candidates": ["away_offense_walks_season_to_date"],
                "home_candidates": ["home_offense_walks_season_to_date"],
                "value_type": "integer",
            },
            {
                "label": "Bases totales acumuladas",
                "away_candidates": ["away_offense_total_bases_season_to_date"],
                "home_candidates": ["home_offense_total_bases_season_to_date"],
                "value_type": "integer",
            },
        ],
    },
    "Últimos 5": {
        "core": [
            {
                "label": "OPS L5",
                "away_candidates": ["away_offense_ops_game_last_5_avg", "away_ops_game_last_5_avg"],
                "home_candidates": ["home_offense_ops_game_last_5_avg", "home_ops_game_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "ISO L5",
                "away_candidates": ["away_offense_iso_game_last_5_avg", "away_iso_game_last_5_avg"],
                "home_candidates": ["home_offense_iso_game_last_5_avg", "home_iso_game_last_5_avg"],
                "value_type": "float",
            },
        ],
        "detail": [
            {
                "label": "Carreras anotadas prom. L5",
                "away_candidates": ["away_offense_runs_scored_last_5_avg", "away_runs_scored_last_5_avg"],
                "home_candidates": ["home_offense_runs_scored_last_5_avg", "home_runs_scored_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "AVG L5",
                "away_candidates": ["away_offense_avg_game_last_5_avg", "away_avg_game_last_5_avg"],
                "home_candidates": ["home_offense_avg_game_last_5_avg", "home_avg_game_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "OBP L5",
                "away_candidates": ["away_offense_obp_game_last_5_avg", "away_obp_game_last_5_avg"],
                "home_candidates": ["home_offense_obp_game_last_5_avg", "home_obp_game_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "SLG L5",
                "away_candidates": ["away_offense_slg_game_last_5_avg", "away_slg_game_last_5_avg"],
                "home_candidates": ["home_offense_slg_game_last_5_avg", "home_slg_game_last_5_avg"],
                "value_type": "float",
            },
        ],
    },
    "Últimos 3": {
        "core": [
            {
                "label": "OPS L3",
                "away_candidates": ["away_offense_ops_game_last_3_avg", "away_ops_game_last_3_avg"],
                "home_candidates": ["home_offense_ops_game_last_3_avg", "home_ops_game_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "ISO L3",
                "away_candidates": ["away_offense_iso_game_last_3_avg", "away_iso_game_last_3_avg"],
                "home_candidates": ["home_offense_iso_game_last_3_avg", "home_iso_game_last_3_avg"],
                "value_type": "float",
            },
        ],
        "detail": [
            {
                "label": "Carreras anotadas prom. L3",
                "away_candidates": ["away_offense_runs_scored_last_3_avg", "away_runs_scored_last_3_avg"],
                "home_candidates": ["home_offense_runs_scored_last_3_avg", "home_runs_scored_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "AVG L3",
                "away_candidates": ["away_offense_avg_game_last_3_avg", "away_avg_game_last_3_avg"],
                "home_candidates": ["home_offense_avg_game_last_3_avg", "home_avg_game_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "OBP L3",
                "away_candidates": ["away_offense_obp_game_last_3_avg", "away_obp_game_last_3_avg"],
                "home_candidates": ["home_offense_obp_game_last_3_avg", "home_obp_game_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "SLG L3",
                "away_candidates": ["away_offense_slg_game_last_3_avg", "away_slg_game_last_3_avg"],
                "home_candidates": ["home_offense_slg_game_last_3_avg", "home_slg_game_last_3_avg"],
                "value_type": "float",
            },
        ],
    },
}

STARTER_METRICS_BY_WINDOW = {
    "Últimos 5": {
        "core": [
            {
                "label": "ERA L5",
                "away_candidates": ["away_starter_era_last_5"],
                "home_candidates": ["home_starter_era_last_5"],
                "value_type": "float",
            },
            {
                "label": "K prom. L5",
                "away_candidates": ["away_starter_strikeouts_last_5_avg"],
                "home_candidates": ["home_starter_strikeouts_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "BB prom. L5",
                "away_candidates": ["away_starter_walks_last_5_avg"],
                "home_candidates": ["home_starter_walks_last_5_avg"],
                "value_type": "float",
            },
        ],
        "detail": [
            {
                "label": "Aperturas previas L5",
                "away_candidates": ["away_starter_starts_count_last_5"],
                "home_candidates": ["home_starter_starts_count_last_5"],
                "value_type": "integer",
            },
            {
                "label": "Outs lanzados prom. L5",
                "away_candidates": ["away_starter_outs_recorded_last_5_avg", "away_starter_innings_pitched_outs_last_5_avg"],
                "home_candidates": ["home_starter_outs_recorded_last_5_avg", "home_starter_innings_pitched_outs_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "Hits permitidos prom. L5",
                "away_candidates": ["away_starter_hits_allowed_last_5_avg"],
                "home_candidates": ["home_starter_hits_allowed_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "Carreras limpias prom. L5",
                "away_candidates": ["away_starter_earned_runs_last_5_avg"],
                "home_candidates": ["home_starter_earned_runs_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "HR permitidos prom. L5",
                "away_candidates": ["away_starter_home_runs_allowed_last_5_avg"],
                "home_candidates": ["home_starter_home_runs_allowed_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "Pitcheos prom. L5",
                "away_candidates": ["away_starter_pitches_thrown_last_5_avg"],
                "home_candidates": ["home_starter_pitches_thrown_last_5_avg"],
                "value_type": "float",
            },
            {
                "label": "Batters faced prom. L5",
                "away_candidates": ["away_starter_batters_faced_last_5_avg"],
                "home_candidates": ["home_starter_batters_faced_last_5_avg"],
                "value_type": "float",
            },
        ],
    },
    "Últimos 3": {
        "core": [
            {
                "label": "ERA L3",
                "away_candidates": ["away_starter_era_last_3"],
                "home_candidates": ["home_starter_era_last_3"],
                "value_type": "float",
            },
            {
                "label": "K prom. L3",
                "away_candidates": ["away_starter_strikeouts_last_3_avg"],
                "home_candidates": ["home_starter_strikeouts_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "BB prom. L3",
                "away_candidates": ["away_starter_walks_last_3_avg"],
                "home_candidates": ["home_starter_walks_last_3_avg"],
                "value_type": "float",
            },
        ],
        "detail": [
            {
                "label": "Aperturas previas L3",
                "away_candidates": ["away_starter_starts_count_last_3"],
                "home_candidates": ["home_starter_starts_count_last_3"],
                "value_type": "integer",
            },
            {
                "label": "Outs lanzados prom. L3",
                "away_candidates": ["away_starter_outs_recorded_last_3_avg", "away_starter_innings_pitched_outs_last_3_avg"],
                "home_candidates": ["home_starter_outs_recorded_last_3_avg", "home_starter_innings_pitched_outs_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "Hits permitidos prom. L3",
                "away_candidates": ["away_starter_hits_allowed_last_3_avg"],
                "home_candidates": ["home_starter_hits_allowed_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "Carreras limpias prom. L3",
                "away_candidates": ["away_starter_earned_runs_last_3_avg"],
                "home_candidates": ["home_starter_earned_runs_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "HR permitidos prom. L3",
                "away_candidates": ["away_starter_home_runs_allowed_last_3_avg"],
                "home_candidates": ["home_starter_home_runs_allowed_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "Pitcheos prom. L3",
                "away_candidates": ["away_starter_pitches_thrown_last_3_avg"],
                "home_candidates": ["home_starter_pitches_thrown_last_3_avg"],
                "value_type": "float",
            },
            {
                "label": "Batters faced prom. L3",
                "away_candidates": ["away_starter_batters_faced_last_3_avg"],
                "home_candidates": ["home_starter_batters_faced_last_3_avg"],
                "value_type": "float",
            },
        ],
    },
}


def find_pregame_features_file() -> Path:
    """
    Busca el archivo final pregame_features_game y usa el mas reciente.
    """
    candidates: dict[Path, Path] = {}

    for base_dir in SEARCH_DIRS:
        if not base_dir.exists():
            continue

        for pattern in DATASET_PATTERNS:
            for path in base_dir.glob(pattern):
                candidates[path.resolve()] = path

    if not candidates:
        raise FileNotFoundError(
            "No se encontro ningun archivo pregame_features_game en las carpetas esperadas."
        )

    return max(candidates.values(), key=lambda path: path.stat().st_mtime)


@st.cache_data
def load_pregame_features() -> tuple[pd.DataFrame, str]:
    """
    Carga el dataset final del proyecto y devuelve tambien la ruta usada.
    """
    file_path = find_pregame_features_file()

    if file_path.suffix == ".parquet":
        df = pd.read_parquet(file_path)
    elif file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")

    return df, str(file_path)


def find_column(df: pd.DataFrame, candidates: list[str], label: str | None = None) -> str | None:
    """
    Devuelve la primera columna existente dentro de una lista de candidatos.
    """
    for col in candidates:
        if col in df.columns:
            return col
    return None


def find_date_column(df: pd.DataFrame) -> str:
    """
    Detecta automaticamente la columna de fecha principal.
    """
    date_col = find_column(df, DATE_COLUMN_CANDIDATES)
    if date_col is None:
        raise KeyError(
            "No encontre una columna de fecha valida en pregame_features_game."
        )
    return date_col


def get_row_value(row: pd.Series, column_name: str | None):
    """
    Lee un valor de una fila de forma segura.
    """
    if column_name is None:
        return pd.NA
    return row.get(column_name, pd.NA)


def format_value(value, value_type: str = "text") -> str:
    """
    Formatea valores para Streamlit sin mostrar None/NaN crudos.
    """
    if pd.isna(value):
        return "N/D"

    if value_type == "flag":
        if value in (1, True):
            return "Si"
        if value in (0, False):
            return "No"

    if value_type == "integer":
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return str(value)

    if value_type == "float":
        try:
            return f"{float(value):.3f}"
        except (TypeError, ValueError):
            return str(value)

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if hasattr(value, "isoformat") and not isinstance(value, str):
        try:
            return value.isoformat()
        except TypeError:
            pass

    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.3f}"

    return str(value)


def inject_theme_styles() -> None:
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, {MLB_BG} 0%, #FFFFFF 22%, {MLB_BG} 100%);
            color: {MLB_TEXT};
        }}
        .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}
        h1, h2, h3 {{
            color: {MLB_BLUE};
            letter-spacing: 0.01em;
        }}
        div[data-testid="stMetric"] {{
            background: white;
            border: 1px solid {MLB_BORDER};
            border-radius: 14px;
            padding: 0.35rem 0.4rem;
            box-shadow: 0 8px 18px rgba(12, 35, 64, 0.05);
        }}
        div[data-testid="stMetricLabel"] {{
            color: #486581;
        }}
        div[data-testid="stMetricValue"] {{
            color: {MLB_BLUE};
        }}
        div[data-testid="stDataFrame"] {{
            border: 1px solid {MLB_BORDER};
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 8px 20px rgba(12, 35, 64, 0.04);
            background: white;
        }}
        .hero-banner {{
            background: linear-gradient(90deg, {MLB_BLUE} 0%, {MLB_BLUE} 76%, {MLB_RED} 100%);
            border-radius: 18px;
            padding: 1.1rem 1.35rem;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 12px 28px rgba(12, 35, 64, 0.18);
        }}
        .hero-banner .eyebrow {{
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            opacity: 0.8;
            margin-bottom: 0.25rem;
        }}
        .hero-banner .title {{
            font-size: 1.5rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .hero-banner .copy {{
            font-size: 0.98rem;
            opacity: 0.92;
        }}
        .section-card {{
            background: white;
            border: 1px solid {MLB_BORDER};
            border-left: 6px solid {MLB_BLUE};
            border-radius: 16px;
            padding: 0.95rem 1.1rem;
            margin: 1rem 0 0.6rem 0;
            box-shadow: 0 10px 24px rgba(12, 35, 64, 0.05);
        }}
        .section-card.red-accent {{
            border-left-color: {MLB_RED};
        }}
        .section-kicker {{
            color: {MLB_RED};
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            margin-bottom: 0.15rem;
        }}
        .section-title {{
            color: {MLB_BLUE};
            font-size: 1.15rem;
            font-weight: 700;
            margin-bottom: 0.2rem;
        }}
        .section-copy {{
            color: #486581;
            font-size: 0.95rem;
            line-height: 1.5;
            margin: 0;
        }}        .compare-card {{
            background: white;
            border: 1px solid {MLB_BORDER};
            border-radius: 16px;
            padding: 0.9rem 1rem;
            margin-bottom: 0.75rem;
            box-shadow: 0 8px 20px rgba(12, 35, 64, 0.05);
        }}
        .compare-card-title {{
            color: {MLB_BLUE};
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            margin-bottom: 0.55rem;
        }}
        .compare-grid {{
            display: grid;
            grid-template-columns: 1fr 120px 1fr;
            gap: 0.75rem;
            align-items: center;
        }}
        .compare-side {{
            background: {MLB_BG};
            border-radius: 12px;
            padding: 0.65rem 0.8rem;
            border: 1px solid {MLB_BORDER};
        }}
        .compare-team-label {{
            color: #486581;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.2rem;
        }}
        .compare-value {{
            color: {MLB_BLUE};
            font-size: 1.2rem;
            font-weight: 700;
            line-height: 1.1;
        }}
        .compare-metric-name {{
            text-align: center;
            color: #486581;
            font-size: 0.85rem;
            font-weight: 600;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(kicker: str, title: str, copy: str, accent: str = "blue") -> None:
    accent_class = " red-accent" if accent == "red" else ""
    st.markdown(
        f"""
        <div class="section-card{accent_class}">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <p class="section-copy">{copy}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_games_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construye una tabla resumida de partidos para la fecha seleccionada.
    """
    result = pd.DataFrame(index=df.index)

    for output_col, candidates, value_type in GAME_TABLE_FIELDS:
        source_col = find_column(df, candidates)
        if source_col is None:
            result[output_col] = "N/D"
            continue

        result[output_col] = df[source_col].apply(lambda value: format_value(value, value_type))

    if "gamePk" not in result.columns:
        raise KeyError("No encontre la columna gamePk en el dataset.")

    return result[
        ["gamePk", "away_team", "home_team", "away_pitcher", "home_pitcher", "status"]
    ].sort_values(
        by=["away_team", "home_team", "gamePk"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def build_flags_table(
    df: pd.DataFrame,
    matchup_row: pd.Series,
    away_label: str,
    home_label: str,
) -> pd.DataFrame:
    """
    Construye una tabla de cobertura de datos solo con flags disponibles.
    """
    rows = []

    for group_name, away_col, home_col in FLAG_FIELDS:
        away_exists = away_col in df.columns
        home_exists = home_col in df.columns

        if not away_exists and not home_exists:
            continue

        rows.append(
            {
                "Grupo": group_name,
                "Flag": away_col.replace("away_", ""),
                away_label: format_value(get_row_value(matchup_row, away_col), "flag"),
                home_label: format_value(get_row_value(matchup_row, home_col), "flag"),
            }
        )

    return pd.DataFrame(rows)


def build_matchup_header(df: pd.DataFrame, matchup_row: pd.Series) -> dict[str, str]:
    """
    Reune los campos base del partido seleccionado.
    """
    values = {}
    for key, candidates in MATCHUP_BASE_FIELDS.items():
        column_name = find_column(df, candidates)
        values[key] = format_value(get_row_value(matchup_row, column_name))
    return values


def build_metric_section(
    df: pd.DataFrame,
    matchup_row: pd.Series,
    metric_definitions: list[dict],
    away_label: str,
    home_label: str,
) -> pd.DataFrame:
    """
    Construye una tabla comparativa away vs home para una seccion especifica.
    """
    rows = []

    for definition in metric_definitions:
        away_col = find_column(df, definition["away_candidates"])
        home_col = find_column(df, definition["home_candidates"])
        away_raw = get_row_value(matchup_row, away_col)
        home_raw = get_row_value(matchup_row, home_col)

        rows.append(
            {
                "Metric": definition["label"],
                away_label: format_value(away_raw, definition["value_type"]),
                home_label: format_value(home_raw, definition["value_type"]),
                "_away_raw": away_raw,
                "_home_raw": home_raw,
                "_highlight": definition.get("highlight"),
            }
        )

    return pd.DataFrame(rows)


def get_run_diff_style(value) -> str:
    if pd.isna(value):
        return ""

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return ""

    if numeric_value > 0:
        return f"color: {MLB_GREEN}; font-weight: 700;"
    if numeric_value < 0:
        return f"color: {MLB_RED}; font-weight: 700;"
    return f"color: {MLB_BLUE}; font-weight: 600;"


def style_standard_table(df: pd.DataFrame):
    return (
        df.style.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [
                        ("background-color", MLB_BLUE),
                        ("color", "white"),
                        ("font-weight", "600"),
                        ("border", f"1px solid {MLB_BLUE}"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border-bottom", f"1px solid {MLB_BORDER}"),
                        ("color", MLB_TEXT),
                    ],
                },
            ]
        )
    )


def style_comparison_table(df: pd.DataFrame):
    visible_columns = [col for col in df.columns if not col.startswith("_")]
    visible_df = df[visible_columns].copy()

    def highlight_rows(row: pd.Series) -> list[str]:
        source_row = df.loc[row.name]
        styles = [""] * len(row)

        if source_row["_highlight"] == "run_diff":
            styles[1] = get_run_diff_style(source_row["_away_raw"])
            styles[2] = get_run_diff_style(source_row["_home_raw"])

        return styles

    return style_standard_table(visible_df).apply(highlight_rows, axis=1)

def render_metric_cards(section_df: pd.DataFrame, away_label: str, home_label: str) -> None:
    """
    Renderiza una lista compacta de tarjetas comparativas away vs home.
    """
    for _, row in section_df.iterrows():
        away_value = row.get(away_label, "N/D")
        home_value = row.get(home_label, "N/D")
        metric_name = row.get("Metric", "")

        st.markdown(
            f"""
            <div class="compare-card">
                <div class="compare-card-title">{metric_name}</div>
                <div class="compare-grid">
                    <div class="compare-side">
                        <div class="compare-team-label">{away_label}</div>
                        <div class="compare-value">{away_value}</div>
                    </div>
                    <div class="compare-metric-name">vs</div>
                    <div class="compare-side">
                        <div class="compare-team-label">{home_label}</div>
                        <div class="compare-value">{home_value}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


inject_theme_styles()

st.title("Baseball Predict")
st.caption("Fase 4 | Streamlit y exploracion")
st.markdown(
    """
    <div class="hero-banner">
        <div class="eyebrow">MLB Pregame Dashboard</div>
        <div class="title">Lectura previa del matchup</div>
        <div class="copy">Contexto reciente de equipos y abridores con una presentacion mas clara, deportiva y sobria.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    df, file_path = load_pregame_features()
except Exception as exc:
    st.error("No pude cargar el dataset final pregame_features_game.")
    st.exception(exc)
    st.stop()

st.subheader("1. Estado del dataset")

col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Filas", len(df))

with col_b:
    st.metric("Columnas", len(df.columns))

with col_c:
    if "gamePk" in df.columns:
        st.metric("gamePk unicos", df["gamePk"].nunique())
    else:
        st.metric("gamePk unicos", "No encontrado")

st.caption(f"Archivo usado: `{file_path}`")

date_col = find_date_column(df)
df[date_col] = pd.to_datetime(df[date_col], errors="coerce").dt.date

available_dates = sorted(df[date_col].dropna().unique())

if not available_dates:
    st.error("No hay fechas disponibles en el dataset.")
    st.stop()

st.subheader("2. Selector de fecha")

selected_date = st.selectbox(
    "Selecciona una fecha",
    options=available_dates,
    index=0,
)

df_day = df[df[date_col] == selected_date].copy()

st.write(f"Partidos encontrados para {selected_date}: **{len(df_day)}**")

if df_day.empty:
    st.warning("No hay partidos para la fecha seleccionada.")
    st.stop()

games_table = build_games_table(df_day)

st.subheader("3. Partidos del dia")
st.dataframe(
    style_standard_table(games_table),
    use_container_width=True,
    hide_index=True,
)

if "gamePk" not in df_day.columns:
    st.error("No encontre la columna gamePk para seleccionar un partido.")
    st.stop()

game_options_df = games_table.copy()
game_options_df["label"] = (
    game_options_df["away_team"]
    + " @ "
    + game_options_df["home_team"]
    + " | gamePk: "
    + game_options_df["gamePk"]
)

game_label_to_pk = dict(zip(game_options_df["label"], game_options_df["gamePk"]))

selected_game_label = st.selectbox(
    "Selecciona un partido",
    options=game_options_df["label"].tolist(),
)

selected_gamepk = pd.to_numeric(game_label_to_pk[selected_game_label], errors="coerce")
matchup_candidates = df_day[pd.to_numeric(df_day["gamePk"], errors="coerce") == selected_gamepk].copy()

if matchup_candidates.empty:
    st.warning("No encontre el gamePk seleccionado.")
    st.stop()

matchup_row = matchup_candidates.iloc[0]
matchup_header = build_matchup_header(df, matchup_row)
away_display_name = matchup_header["away_team"]
home_display_name = matchup_header["home_team"]

st.subheader("4. Resumen del matchup")

col1, col2 = st.columns(2)

with col1:
    render_section_header(
        "Matchup",
        "Equipos",
        "Vista rapida del partido seleccionado y su identificacion base.",
    )
    st.write(f"**Away:** {away_display_name}")
    st.write(f"**Home:** {home_display_name}")
    st.write(f"**gamePk:** {format_value(matchup_row['gamePk'], 'integer')}")
    st.write(f"**Fecha:** {format_value(matchup_row[date_col])}")

with col2:
    render_section_header(
        "Pregame",
        "Contexto base",
        "Lectura inicial del estado del juego y los abridores probables.",
        accent="red",
    )
    st.write(f"**Pitcher probable away:** {matchup_header['away_pitcher']}")
    st.write(f"**Pitcher probable home:** {matchup_header['home_pitcher']}")
    st.write(f"**Estado:** {matchup_header['status']}")

st.subheader("5. Flags de cobertura de datos")

flags_df = build_flags_table(df, matchup_row, away_display_name, home_display_name)

if flags_df.empty:
    st.info("No encontre columnas de flags de disponibilidad o usabilidad en este dataset.")
else:
    st.dataframe(style_standard_table(flags_df), use_container_width=True, hide_index=True)

st.subheader("6. Matchup pregame")

st.markdown("### Ventanas de comparacion")

col_window_1, col_window_2 = st.columns(2)

with col_window_1:
    selected_offense_window = st.radio(
        "Ofensiva",
        options=["Temporada", "Últimos 5", "Últimos 3"],
        horizontal=True,
        index=1,
    )

with col_window_2:
    selected_starter_window = st.radio(
        "Abridor",
        options=["Últimos 5", "Últimos 3"],
        horizontal=True,
        index=0,
    )

context_df = build_metric_section(df, matchup_row, CONTEXT_METRICS, away_display_name, home_display_name)

offense_core_df = build_metric_section(
    df,
    matchup_row,
    OFFENSE_METRICS_BY_WINDOW[selected_offense_window]["core"],
    away_display_name,
    home_display_name,
)

offense_detail_df = build_metric_section(
    df,
    matchup_row,
    OFFENSE_METRICS_BY_WINDOW[selected_offense_window]["detail"],
    away_display_name,
    home_display_name,
)

starter_core_df = build_metric_section(
    df,
    matchup_row,
    STARTER_METRICS_BY_WINDOW[selected_starter_window]["core"],
    away_display_name,
    home_display_name,
)

starter_detail_df = build_metric_section(
    df,
    matchup_row,
    STARTER_METRICS_BY_WINDOW[selected_starter_window]["detail"],
    away_display_name,
    home_display_name,
)

render_section_header(
    "6.1",
    "Contexto general",
    "Cobertura de datos disponible para el partido y disponibilidad reciente de equipos y abridores.",
)
st.dataframe(style_comparison_table(context_df), use_container_width=True, hide_index=True)

render_section_header(
    "6.2",
    f"Ofensiva | {selected_offense_window}",
    "Comparacion principal de produccion y poder ofensivo segun la ventana seleccionada.",
)
render_metric_cards(offense_core_df, away_display_name, home_display_name)

with st.expander(f"Ver detalle ofensivo | {selected_offense_window}"):
    st.dataframe(
        style_comparison_table(offense_detail_df),
        use_container_width=True,
        hide_index=True,
    )

render_section_header(
    "6.3",
    f"Abridor | {selected_starter_window}",
    "Comparacion principal del abridor probable segun la ventana seleccionada.",
    accent="red",
)
render_metric_cards(starter_core_df, away_display_name, home_display_name)

with st.expander(f"Ver detalle del abridor | {selected_starter_window}"):
    st.dataframe(
        style_comparison_table(starter_detail_df),
        use_container_width=True,
        hide_index=True,
    )