
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / "data/raw/pqrs_sdg_2022_2026.csv"
SALIDA = BASE / "data/processed"
REPORTES = BASE / "reports/metricas"

SALIDA.mkdir(parents=True, exist_ok=True)
REPORTES.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ENTRADA,
    sep=";",
    encoding="utf-8-sig",
    low_memory=False
)

# Fechas originales
for columna in ["RADI_FECH_RADI", "FECHA_RESPUESTA"]:
    df[columna] = pd.to_datetime(
        df[columna],
        errors="coerce",
        dayfirst=True
    )

# Plazos generales según tipo de petición.
# No asignar un plazo general a categorías especiales.
plazos = {
    "DERECHO DE PETICION EN INTERES GENERAL": 15,
    "DERECHO DE PETICION EN INTERES PARTICULAR": 15,
    "DERECHO DE PETICION DE INFORMACIONES": 10,
    "DERECHO DE PETICION DE CONSULTAS": 30
}

df["PLAZO_SLA"] = df["TIPO_PETICION"].map(plazos)

# Normalizar fechas sin hora
inicio = df["RADI_FECH_RADI"].dt.normalize()
fin = df["FECHA_RESPUESTA"].dt.normalize()

# Calendario de festivos de Colombia
fechas_validas = pd.concat([inicio, fin]).dropna()

anio_min = int(fechas_validas.dt.year.min())
anio_max = int(fechas_validas.dt.year.max())

festivos_colombia = holidays.country_holidays(
    "CO",
    years=range(anio_min, anio_max + 1)
)

festivos = np.array(
    list(festivos_colombia.keys()),
    dtype="datetime64[D]"
)

# Identificar registros con fechas coherentes
fechas_coherentes = (
    inicio.notna()
    & fin.notna()
    & (fin >= inicio)
)

df["DIAS_HABILES"] = np.nan

# Contar días hábiles posteriores a la radicación
# hasta la fecha de respuesta inclusive.
# Esta convención es provisional y debe validarse.
indices = df.index[fechas_coherentes]

if len(indices) > 0:
    fechas_inicio = (
        inicio.loc[indices].to_numpy(dtype="datetime64[D]")
    )
    fechas_fin = (
        fin.loc[indices].to_numpy(dtype="datetime64[D]")
    )

    df.loc[indices, "DIAS_HABILES"] = np.busday_count(
        fechas_inicio + np.timedelta64(1, "D"),
        fechas_fin + np.timedelta64(1, "D"),
        holidays=festivos
    )

# Estado de evaluación
df["ESTADO_EVALUACION"] = "PENDIENTE_REVISION"

df.loc[
    df["PLAZO_SLA"].isna(),
    "ESTADO_EVALUACION"
] = "TIPO_NO_EVALUABLE"

df.loc[
    df["PLAZO_SLA"].notna() & fin.isna(),
    "ESTADO_EVALUACION"
] = "SIN_FECHA_RESPUESTA"

df.loc[
    df["PLAZO_SLA"].notna()
    & inicio.notna()
    & fin.notna()
    & (fin < inicio),
    "ESTADO_EVALUACION"
] = "FECHAS_INCONSISTENTES"

evaluables = (
    df["PLAZO_SLA"].notna()
    & fechas_coherentes
)

df.loc[
    evaluables,
    "ESTADO_EVALUACION"
] = "EVALUACION_PRELIMINAR"

# Variable provisional: NO es aún etiqueta validada
df["SUPERA_PLAZO_PRELIMINAR"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int64"
)

df.loc[
    evaluables,
    "SUPERA_PLAZO_PRELIMINAR"
] = (
    df.loc[evaluables, "DIAS_HABILES"]
    > df.loc[evaluables, "PLAZO_SLA"]
).astype("int64")

# Guardar datos sin alterar el CSV original
df.to_csv(
    SALIDA / "pqrs_sla_preliminar.csv",
    index=False,
    encoding="utf-8-sig"
)

resumen = (
    df.groupby("ESTADO_EVALUACION", dropna=False)
    .size()
    .reset_index(name="REGISTROS")
)

resumen.to_csv(
    REPORTES / "resumen_sla_preliminar.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== DIAGNOSTICO PRELIMINAR SLA ===")
print(f"Total de registros: {len(df)}")
print("\nEstado de evaluacion:")
print(resumen.to_string(index=False))

print("\nCasos que superan el plazo nominal:")
print(
    df.loc[evaluables, "SUPERA_PLAZO_PRELIMINAR"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nArchivos generados correctamente.")
print("IMPORTANTE: Las etiquetas NO estan validadas.")