import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.carga_datos import cargar_datos_historicos, COLUMNA_ID, COLUMNA_TARGET

SEMILLA_ALEATORIA = 42
PROPORCION_TEST = 0.20

df = cargar_datos_historicos()

X = df.drop(columns=[COLUMNA_ID, COLUMNA_TARGET])
y = df[COLUMNA_TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=PROPORCION_TEST,
    random_state=SEMILLA_ALEATORIA,
    stratify=y,
)

if __name__ == "__main__":
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")
    print(f"Proporción de test: {PROPORCION_TEST} | Semilla: {SEMILLA_ALEATORIA}")
    print()
    print("Proporción de Churn por conjunto:")
    print(
        pd.DataFrame(
            {
                "completo": y.value_counts(normalize=True),
                "train": y_train.value_counts(normalize=True),
                "test": y_test.value_counts(normalize=True),
            }
        ).round(4)
    )
