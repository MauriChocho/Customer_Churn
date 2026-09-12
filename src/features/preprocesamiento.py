from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.particion import X_train

# TotalCharges es la única con faltantes, y coinciden exactamente con tenure=0
# (clientes nuevos sin facturación acumulada aún): imputar con 0 es correcto
# para las cuatro columnas numéricas, ya que ninguna otra tiene valores nulos.
COLUMNAS_NUMERICAS = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
COLUMNAS_CATEGORICAS = [columna for columna in X_train.columns if columna not in COLUMNAS_NUMERICAS]

pipeline_numerico = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="constant", fill_value=0)),
        ("escalado", StandardScaler()),
    ]
)

pipeline_categorico = Pipeline(
    steps=[
        ("imputacion", SimpleImputer(strategy="most_frequent")),
        ("encoding", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocesador = ColumnTransformer(
    transformers=[
        ("numericas", pipeline_numerico, COLUMNAS_NUMERICAS),
        ("categoricas", pipeline_categorico, COLUMNAS_CATEGORICAS),
    ]
)

if __name__ == "__main__":
    X_train_transformado = preprocesador.fit_transform(X_train)
    print(f"Columnas numéricas: {len(COLUMNAS_NUMERICAS)} | Columnas categóricas: {len(COLUMNAS_CATEGORICAS)}")
    print(f"Columnas de entrada: {X_train.shape[1]} -> columnas luego de preprocesar: {X_train_transformado.shape[1]}")
    print("Primeros nombres de columnas de salida:", list(preprocesador.get_feature_names_out()[:6]))
