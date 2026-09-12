import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.carga_datos import COLUMNA_TARGET

SEMILLA_ALEATORIA = 42
PROPORCION_TEST = 0.30


def particionar_train_test(
    df: pd.DataFrame,
    proporcion_test: float = PROPORCION_TEST,
    semilla_aleatoria: int = SEMILLA_ALEATORIA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Particiona el dataset histórico en train/test de forma reproducible.

    Estratifica por Churn para preservar en ambos subconjuntos la proporción
    de clases del dataset completo (~26% de churn).
    """
    return train_test_split(
        df,
        test_size=proporcion_test,
        random_state=semilla_aleatoria,
        stratify=df[COLUMNA_TARGET],
    )
