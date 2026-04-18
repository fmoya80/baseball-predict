import pandas as pd

from src.pipeline_paths import PREGAME_FEATURES_MASTER_FILE


def load_pregame_features_master() -> pd.DataFrame:
    """
    Carga el archivo maestro oficial del pipeline pregame.
    """
    if not PREGAME_FEATURES_MASTER_FILE.exists():
        raise FileNotFoundError(
            f"No existe el master oficial en: {PREGAME_FEATURES_MASTER_FILE}"
        )

    df = pd.read_parquet(PREGAME_FEATURES_MASTER_FILE)
    return df