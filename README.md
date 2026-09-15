# Sistema de Predicción de Abandono de Clientes (Customer Churn)

Proyecto Integrador — Laboratorio de Minería de Datos — ISTEA (2do cuatrimestre 2026)
Profesor: Diego Mosquera

## 1. Problema de negocio

Una empresa de telecomunicaciones necesita identificar clientes con riesgo de abandonar el
servicio (`Churn`). El sistema debe estimar la probabilidad de abandono de un cliente y
devolver un nivel de riesgo (`LOW` / `MEDIUM` / `HIGH`) consumible por otros sistemas
(por ejemplo, un CRM o una campaña de retención).

Este repositorio evoluciona en tres entregas incrementales:

| Entrega | Hito técnico |
|---|---|
| 1 — Primer Parcial | Proyecto reproducible: Git + DVC + MLflow + Model Registry |
| 2 — Segundo Parcial | Servicio de inferencia: testing + FastAPI + Docker + Compose |
| 3 — Examen Final | Solución productiva académica: CI/CD + GHCR + Linux + monitoring + video |

## 2. Dataset

Provisto por la cátedra en `data/` (ver [README_DATOS.md](README_DATOS.md) para el detalle
completo). Resumen:

- `data/raw/customer_churn_historical.csv`: histórico con la variable objetivo `Churn`, usado
  para entrenamiento y evaluación.
- `data/production/customer_churn_current.csv`: lote sin `Churn`, reservado para la etapa de
  monitoreo y análisis de drift.
- `data/scoring/scoring_batch.csv`: lote pequeño sin target, para pruebas de inferencia.
- `metadata/`: diccionario de datos y esquema.
- `examples/`: ejemplos de payloads válidos e inválidos para `POST /predict`.

`customerID` es un identificador y no se utiliza como variable predictora.

## 3. Estructura del repositorio

```
ISTEA_Proyecto_Churn_Datos/
├── data/                 # Datos versionados con DVC (raw, production, scoring)
├── notebooks/            # Exploración y análisis (no forman parte del flujo productivo)
├── src/
│   ├── data/             # Carga y partición de datos
│   ├── features/         # Preprocesamiento (Pipeline, ColumnTransformer)
│   ├── training/         # Entrenamiento de modelos
│   ├── evaluation/       # Métricas y comparación de modelos
│   └── inference/        # Lógica de predicción reutilizada por la API
├── app/                  # Servicio FastAPI (/health, /predict)
├── tests/                # Suite de pytest
├── monitoring/           # Análisis de Data Drift con Evidently
├── models/               # Artefactos de modelo (no versionados en Git)
├── scripts/              # Scripts ejecutables (entrenamiento, scoring batch, etc.)
├── .github/workflows/    # CI/CD con GitHub Actions
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── .gitignore 
├── .dockerignore
└── README.md
```

## 4. Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 5. Configuración de acceso a los datos (DVC + DagsHub)

Los datos del proyecto están versionados con DVC y alojados en DagsHub, no en este repositorio.

Para poder descargarlos, cada persona debe configurar su propio acceso:

1. Crear una cuenta en DagsHub (si no la tenés).
2. Generar un token personal desde Settings → Tokens.
3. Configurar la autenticación local (reemplazando por tus propios datos):

   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user <tu_usuario_dagshub>
   dvc remote modify origin --local password <tu_token>

4. Descargar los datos:

   dvc pull


## 6. Estado actual

Proyecto en etapa de Entrega 1 (EDA, pipeline de preprocesamiento, comparación de modelos,
versionado de datos con DVC y registro de experimentos con MLflow). Las secciones de
entrenamiento, MLflow/DagsHub, testing, ejecución con Docker Compose, CI/CD, deployment y
monitoreo se completarán progresivamente en este README a medida que avancen las entregas.
