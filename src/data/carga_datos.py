from pathlib import Path

import pandas as pd

RAIZ_PROYECTO = Path(__file__).resolve().parents[2]

RUTA_DATOS_HISTORICOS = RAIZ_PROYECTO / "data" / "raw" / "customer_churn_historical.csv"
RUTA_DATOS_PRODUCCION = RAIZ_PROYECTO / "data" / "production" / "customer_churn_current.csv"
RUTA_DATOS_SCORING = RAIZ_PROYECTO / "data" / "scoring" / "scoring_batch.csv"

COLUMNA_ID = "customerID"
COLUMNA_TARGET = "Churn"


def cargar_datos_historicos(ruta: Path = RUTA_DATOS_HISTORICOS) -> pd.DataFrame:
    """Carga el dataset histórico, que incluye la variable objetivo Churn."""
    return pd.read_csv(ruta)


def cargar_datos_produccion(ruta: Path = RUTA_DATOS_PRODUCCION) -> pd.DataFrame:
    """Carga el lote de producción simulado (sin Churn), reservado para drift."""
    return pd.read_csv(ruta)


def cargar_datos_scoring(ruta: Path = RUTA_DATOS_SCORING) -> pd.DataFrame:
    """Carga el lote pequeño de scoring, sin variable objetivo."""
    return pd.read_csv(ruta)
