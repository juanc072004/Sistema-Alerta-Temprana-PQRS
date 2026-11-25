
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / "data/raw/pqrs_sdg_2022_2026.csv"
SALIDA = BASE / "reports/metricas"

SALIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ENTRADA,
    sep=";",
    encoding="utf-8-sig",
    low_memory=False
)

columnas = [
    "FECHA_REGISTRO",
    "RADI_FECH_RADI",
    "FECHA_RESPUESTA",
    "TIEMPO_TRAMITE",
    "TIPO_PETICION"
]

faltantes = [c for c in columnas if c not in df.columns]

if faltantes:
    raise ValueError(f"Faltan columnas: {faltantes}")

for columna in [
    "FECHA_REGISTRO",
    "RADI_FECH_RADI",
    "FECHA_RESPUESTA"
]:
    df[columna] = pd.to_datetime(
        df[columna],
        errors="coerce",
        dayfirst=True
    )

df["DIAS_DESDE_REGISTRO"] = (
    df["FECHA_RESPUESTA"] - df["FECHA_REGISTRO"]
).dt.days

df["DIAS_DESDE_RADICACION"] = (
    df["FECHA_RESPUESTA"] - df["RADI_FECH_RADI"]
).dt.days

df["DIAS_RADICACION_VS_REGISTRO"] = (
    df["RADI_FECH_RADI"] - df["FECHA_REGISTRO"]
).dt.days

indicadores = {
    "total_registros": len(df),
    "sin_fecha_radicacion": int(
        df["RADI_FECH_RADI"].isna().sum()
    ),
    "sin_fecha_respuesta": int(
        df["FECHA_RESPUESTA"].isna().sum()
    ),
    "negativos_desde_registro": int(
        (df["DIAS_DESDE_REGISTRO"] < 0).sum()
    ),
    "negativos_desde_radicacion": int(
        (df["DIAS_DESDE_RADICACION"] < 0).sum()
    ),
    "radicacion_anterior_registro": int(
        (df["DIAS_RADICACION_VS_REGISTRO"] < 0).sum()
    ),
    "mediana_dias_desde_registro":
        df["DIAS_DESDE_REGISTRO"].median(),
    "mediana_dias_desde_radicacion":
        df["DIAS_DESDE_RADICACION"].median()
}

print("\n=== DIAGNOSTICO DE FECHAS PQRS ===")

for nombre, valor in indicadores.items():
    print(f"{nombre}: {valor}")

pd.DataFrame(
    [indicadores]
).to_csv(
    SALIDA / "diagnostico_fechas.csv",
    index=False,
    encoding="utf-8-sig"
)

reporte = (
    df.groupby("TIPO_PETICION", dropna=False)
    .agg(
        registros=("TIPO_PETICION", "size"),
        mediana_desde_registro=(
            "DIAS_DESDE_REGISTRO", "median"
        ),
        mediana_desde_radicacion=(
            "DIAS_DESDE_RADICACION", "median"
        ),
        negativos_registro=(
            "DIAS_DESDE_REGISTRO",
            lambda x: int((x < 0).sum())
        ),
        negativos_radicacion=(
            "DIAS_DESDE_RADICACION",
            lambda x: int((x < 0).sum())
        )
    )
    .reset_index()
)

reporte.to_csv(
    SALIDA / "diagnostico_fechas_por_tipo.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nReportes generados correctamente.")