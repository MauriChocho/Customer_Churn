from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from src.data.particion import X_test, X_train, y_test, y_train
from src.features.preprocesamiento import preprocesador

NOMBRE_EXPERIMENTO = "customer-churn"
CLASE_POSITIVA = "Yes"

# Se usa una ruta corta fuera del proyecto porque la ruta real del repo (muy
# anidada) supera el límite de 260 caracteres de Windows al guardar artefactos
# del modelo. Cuando se configure DagsHub, esto se reemplaza por su URI remota.
RUTA_MLFLOW_LOCAL = Path.home() / "mlflow_local" / "customer-churn"
RUTA_MLFLOW_LOCAL.mkdir(parents=True, exist_ok=True)
mlflow.set_tracking_uri(f"sqlite:///{RUTA_MLFLOW_LOCAL / 'mlflow.db'}")

if mlflow.get_experiment_by_name(NOMBRE_EXPERIMENTO) is None:
    mlflow.create_experiment(
        NOMBRE_EXPERIMENTO,
        artifact_location=(RUTA_MLFLOW_LOCAL / "artifacts").as_uri(),
    )
mlflow.set_experiment(NOMBRE_EXPERIMENTO)


def entrenar_y_registrar(nombre_run: str, modelo) -> dict:
    """Entrena un Pipeline (preprocesamiento + modelo), lo evalúa sobre test
    y registra parámetros, métricas, matriz de confusión y el modelo en MLflow."""
    pipeline_completo = Pipeline(
        steps=[
            ("preprocesamiento", preprocesador),
            ("modelo", modelo),
        ]
    )

    with mlflow.start_run(run_name=nombre_run):
        pipeline_completo.fit(X_train, y_train)

        y_pred = pipeline_completo.predict(X_test)
        indice_positivo = list(pipeline_completo.classes_).index(CLASE_POSITIVA)
        y_proba = pipeline_completo.predict_proba(X_test)[:, indice_positivo]

        metricas = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, pos_label=CLASE_POSITIVA, zero_division=0),
            "recall": recall_score(y_test, y_pred, pos_label=CLASE_POSITIVA, zero_division=0),
            "f1": f1_score(y_test, y_pred, pos_label=CLASE_POSITIVA, zero_division=0),
            "roc_auc": roc_auc_score(y_test == CLASE_POSITIVA, y_proba),
        }

        mlflow.log_param("modelo", type(modelo).__name__)
        mlflow.log_params(modelo.get_params())
        mlflow.log_metrics(metricas)

        figura_matriz, eje = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=eje)
        eje.set_title(nombre_run)
        mlflow.log_figure(figura_matriz, "matriz_confusion.png")
        plt.close(figura_matriz)

        mlflow.sklearn.log_model(
            pipeline_completo,
            name="modelo",
            input_example=X_train.head(3),
            serialization_format="cloudpickle",
        )

    print(f"{nombre_run:35s} " + " | ".join(f"{clave}={valor:.4f}" for clave, valor in metricas.items()))
    return metricas


if __name__ == "__main__":
    entrenar_y_registrar("baseline_most_frequent", DummyClassifier(strategy="most_frequent"))

    entrenar_y_registrar(
        "logistic_regression_default",
        LogisticRegression(max_iter=1000, random_state=42),
    )
    entrenar_y_registrar(
        "logistic_regression_balanced",
        LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    )

    entrenar_y_registrar(
        "random_forest_default",
        RandomForestClassifier(n_estimators=200, random_state=42),
    )
    entrenar_y_registrar(
        "random_forest_balanced",
        RandomForestClassifier(n_estimators=200, random_state=42, class_weight="balanced"),
    )
    entrenar_y_registrar(
        "random_forest_balanced_profundo",
        RandomForestClassifier(n_estimators=400, max_depth=10, random_state=42, class_weight="balanced"),
    )

    runs = mlflow.search_runs(experiment_names=[NOMBRE_EXPERIMENTO], order_by=["metrics.f1 DESC"])

    columnas_metricas = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    tabla_comparativa = runs[["tags.mlflow.runName"] + [f"metrics.{m}" for m in columnas_metricas]].copy()
    tabla_comparativa.columns = ["run"] + columnas_metricas
    print("\nComparación de runs (ordenado por F1):")
    print(tabla_comparativa.round(4).to_string(index=False))

    # Selección justificada: dado que un falso negativo (cliente que iba a
    # abandonar y no se detecta) es el error costoso para el negocio, F1
    # (equilibrio precision/recall) es la métrica principal para elegir el
    # modelo candidato, por encima de accuracy.
    mejor_run = runs.iloc[0]
    print(
        f"\nMejor run por F1: {mejor_run['tags.mlflow.runName']} "
        f"(F1={mejor_run['metrics.f1']:.4f}, run_id={mejor_run['run_id']})"
    )

    modelo_registrado = mlflow.register_model(
        model_uri=f"runs:/{mejor_run['run_id']}/modelo",
        name="churn-model",
    )
    print(f"Modelo registrado: {modelo_registrado.name} v{modelo_registrado.version} (origen: run {mejor_run['run_id']})")
