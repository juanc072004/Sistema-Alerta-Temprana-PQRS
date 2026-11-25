
from pathlib import Path
import pandas as pd
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RUTA = BASE / "data/raw/pqrs_sdg_2022_2026.csv"
SALIDA = BASE / "reports/metricas"
SALIDA.mkdir(parents=True, exist_ok=True)

if not RUTA.exists():
    raise FileNotFoundError(f"No existe el CSV: {RUTA}")

df = pd.read_csv(
    RUTA,
    sep=";",
    encoding="utf-8-sig",
    low_memory=False
)

columnas = [
    "FECHA_REGISTRO",
    "FECHA_RESPUESTA",
    "TIEMPO_TRAMITE",
    "TIPO_PETICION"
]

faltantes = [c for c in columnas if c not in df.columns]
if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")

for col in ["FECHA_REGISTRO", "FECHA_RESPUESTA"]:
    df[col] = pd.to_datetime(
        df[col], errors="coerce", dayfirst=True
    )

df["TIEMPO_NUM"] = pd.to_numeric(
    df["TIEMPO_TRAMITE"], errors="coerce"
)

df["DIAS_CALENDARIO"] = (
    df["FECHA_RESPUESTA"] - df["FECHA_REGISTRO"]
).dt.days

validos = df.dropna(
    subset=["DIAS_CALENDARIO", "TIEMPO_NUM"]
).copy()

validos["DIFERENCIA"] = (
    validos["TIEMPO_NUM"] - validos["DIAS_CALENDARIO"]
)

resumen = {
    "registros_totales": len(df),
    "comparables": len(validos),
    "fechas_negativas": int(
        (df["DIAS_CALENDARIO"] < 0).sum()
    ),
    "tiempos_negativos": int(
        (df["TIEMPO_NUM"] < 0).sum()
    ),
    "coincidencias_exactas": int(
        np.isclose(validos["DIFERENCIA"], 0).sum()
    ),
    "diferencia_mediana": validos["DIFERENCIA"].median()
}

print("\nDIAGNOSTICO DE TIEMPOS")
for clave, valor in resumen.items():
    print(f"{clave}: {valor}")

reporte = (
    validos.groupby("TIPO_PETICION", dropna=False)
    .agg(
        registros=("DIFERENCIA", "size"),
        mediana_dias_calendario=("DIAS_CALENDARIO", "median"),
        mediana_tiempo_tramite=("TIEMPO_NUM", "median"),
        mediana_diferencia=("DIFERENCIA", "median")
    )
    .reset_index()
)

reporte.to_csv(
    SALIDA / "verificacion_tiempos.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nReporte guardado correctamente.")