
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBClassifier


BASE = Path(__file__).resolve().parent.parent

ENTRADA = (
    BASE / "data/processed/pqrs_sla_preliminar.csv"
)

MODELOS = BASE / "models"
REPORTES = BASE / "reports/metricas"

MODELOS.mkdir(parents=True, exist_ok=True)
REPORTES.mkdir(parents=True, exist_ok=True)


# 1. CARGAR DATOS REALES

df = pd.read_csv(
    ENTRADA,
    encoding="utf-8-sig",
    low_memory=False
)

df["RADI_FECH_RADI"] = pd.to_datetime(
    df["RADI_FECH_RADI"],
    errors="coerce"
)


# 2. SELECCIONAR CASOS PARA EL EXPERIMENTO

datos = df.loc[
    (df["ESTADO_EVALUACION"] == "EVALUACION_PRELIMINAR")
    & (
        df["ESTADO_ORFEO"]
        .astype(str)
        .str.strip()
        .str.upper() == "TRAMITE CERRADO"
    )
    & df["SUPERA_PLAZO_PRELIMINAR"].notna()
    & df["RADI_FECH_RADI"].notna()
].copy()

# Eliminar posibles filas repetidas
datos = datos.drop_duplicates()

# Variables conocidas desde la radicación
datos["ANIO_RADICACION"] = (
    datos["RADI_FECH_RADI"].dt.year
)

datos["MES_RADICACION"] = (
    datos["RADI_FECH_RADI"].dt.month
)

datos["DIA_SEMANA_RADICACION"] = (
    datos["RADI_FECH_RADI"].dt.dayofweek
)

variables_categoricas = [
    "TIPO_PETICION",
    "MEDIO_RECEPCION",
    "NOMBRE_LOCALIDAD",
    "SUBTEMA"
]

variables_numericas = [
    "ANIO_RADICACION",
    "MES_RADICACION",
    "DIA_SEMANA_RADICACION"
]

variables = (
    variables_categoricas + variables_numericas
)

# Comprobar columnas
faltantes = [
    c for c in variables if c not in datos.columns
]

if faltantes:
    raise ValueError(
        f"Faltan columnas: {faltantes}"
    )

datos = datos.sort_values(
    "RADI_FECH_RADI"
).reset_index(drop=True)

X = datos[variables].copy()

for columna in variables_categoricas:
    X[columna] = (
        X[columna]
        .astype("string")
        .fillna("SIN_DATO")
        .astype(str)
    )

for columna in variables_numericas:
    X[columna] = pd.to_numeric(
        X[columna],
        errors="coerce"
    )

y = (
    datos["SUPERA_PLAZO_PRELIMINAR"]
    .astype(int)
)


# 3. DIVISION TEMPORAL 80/20

# Evita compartir una misma fecha entre
# entrenamiento y prueba.
fechas = np.sort(
    datos["RADI_FECH_RADI"]
    .dt.normalize()
    .unique()
)

if len(fechas) < 5:
    raise ValueError(
        "No hay suficientes fechas distintas."
    )

posicion = int(len(fechas) * 0.8)
fecha_corte = fechas[posicion]

entrenamiento = (
    datos["RADI_FECH_RADI"]
    .dt.normalize() < fecha_corte
)

prueba = ~entrenamiento

X_train = X.loc[entrenamiento]
X_test = X.loc[prueba]

y_train = y.loc[entrenamiento]
y_test = y.loc[prueba]

if (
    y_train.nunique() < 2
    or y_test.nunique() < 2
):
    raise ValueError(
        "La división temporal no contiene "
        "ambas clases en entrenamiento y prueba."
    )

print("\n=== DISTRIBUCION TEMPORAL ===")
print(f"Fecha de corte: {fecha_corte}")
print(f"Entrenamiento: {len(X_train)}")
print(f"Prueba: {len(X_test)}")

print("\nClases entrenamiento:")
print(y_train.value_counts())

print("\nClases prueba:")
print(y_test.value_counts())


# 4. PREPROCESAMIENTO

def crear_preprocesador():
    return ColumnTransformer(
        transformers=[
            (
                "categoricas",
                Pipeline([
                    (
                        "imputar",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "codificar",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                variables_categoricas
            ),
            (
                "numericas",
                SimpleImputer(
                    strategy="median"
                ),
                variables_numericas
            )
        ]
    )


# 5. MODELOS

positivos = int(y_train.sum())
negativos = int(len(y_train) - positivos)

peso_positivo = negativos / positivos

modelos = {
    "random_forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),
    "xgboost": XGBClassifier(
        n_estimators=250,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=peso_positivo,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
        n_jobs=4
    )
}


# 6. ENTRENAR Y EVALUAR

resultados = []

for nombre, clasificador in modelos.items():

    print(f"\n=== ENTRENANDO {nombre.upper()} ===")

    pipeline = Pipeline([
        ("preprocesador", crear_preprocesador()),
        ("modelo", clasificador)
    ])

    pipeline.fit(X_train, y_train)

    probabilidades = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Umbral inicial, sin optimización sobre prueba
    predicciones = (
        probabilidades >= 0.5
    ).astype(int)

    pr_auc = average_precision_score(
        y_test,
        probabilidades
    )

    roc_auc = roc_auc_score(
        y_test,
        probabilidades
    )

    matriz = confusion_matrix(
        y_test,
        predicciones,
        labels=[0, 1]
    )

    print("\nReporte de clasificación:")
    print(
        classification_report(
            y_test,
            predicciones,
            zero_division=0
        )
    )

    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print("\nMatriz de confusión:")
    print(matriz)

    resultados.append({
        "modelo": nombre,
        "registros_entrenamiento": len(X_train),
        "registros_prueba": len(X_test),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "verdaderos_negativos": int(matriz[0, 0]),
        "falsos_positivos": int(matriz[0, 1]),
        "falsos_negativos": int(matriz[1, 0]),
        "verdaderos_positivos": int(matriz[1, 1])
    })

    joblib.dump(
        pipeline,
        MODELOS / f"{nombre}_experimental.joblib"
    )


# 7. GUARDAR RESULTADOS

tabla = pd.DataFrame(resultados)

tabla.to_csv(
    REPORTES / "comparacion_modelos_experimentales.csv",
    index=False,
    encoding="utf-8-sig"
)

metadata = {
    "objetivo": "SUPERA_PLAZO_PRELIMINAR",
    "alcance": "Modelo experimental",
    "advertencia": (
        "No representa incumplimiento legal confirmado."
    ),
    "variables": variables,
    "fecha_corte": str(fecha_corte),
    "total_entrenamiento": len(X_train),
    "total_prueba": len(X_test)
}

with open(
    REPORTES / "metadata_experimento.json",
    "w",
    encoding="utf-8"
) as archivo:
    json.dump(
        metadata,
        archivo,
        indent=4,
        ensure_ascii=False
    )

print("\n=== COMPARACION FINAL ===")
print(tabla.to_string(index=False))

print("\nEntrenamiento experimental terminado.")
print("Modelos guardados en models/")
print("Resultados guardados en reports/metricas/")