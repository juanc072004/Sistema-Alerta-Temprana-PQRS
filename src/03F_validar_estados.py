
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
ENTRADA = BASE / "data/processed/pqrs_sla_preliminar.csv"
SALIDA = BASE / "reports/metricas"
SALIDA.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(
    ENTRADA,
    encoding="utf-8-sig",
    low_memory=False
)

# Normalizar estado sin modificar el dato original
df["ESTADO_NORMALIZADO"] = (
    df["ESTADO_ORFEO"]
    .fillna("SIN_ESTADO")
    .astype(str)
    .str.strip()
    .str.upper()
)

# Comprobar si existe radicado de respuesta
df["TIENE_RADICADO_RESPUESTA"] = (
    df["RADICADO_RESPUESTA"].notna()
    & df["RADICADO_RESPUESTA"]
        .astype(str)
        .str.strip()
        .ne("")
)

# Frecuencia de estados
estados = (
    df.groupby("ESTADO_NORMALIZADO", dropna=False)
    .size()
    .reset_index(name="REGISTROS")
    .sort_values("REGISTROS", ascending=False)
)

estados.to_csv(
    SALIDA / "estados_orfeo.csv",
    index=False,
    encoding="utf-8-sig"
)

# Relación entre estado y clasificación preliminar
cruce = (
    df.groupby(
        [
            "ESTADO_NORMALIZADO",
            "ESTADO_EVALUACION",
            "TIENE_RADICADO_RESPUESTA"
        ],
        dropna=False
    )
    .size()
    .reset_index(name="REGISTROS")
)

cruce.to_csv(
    SALIDA / "cruce_estados_sla.csv",
    index=False,
    encoding="utf-8-sig"
)

# Resumen específico de los registros evaluables
evaluables = df[
    df["ESTADO_EVALUACION"]
    == "EVALUACION_PRELIMINAR"
]

print("\n=== ESTADOS ORFEO ===")
print(estados.head(20).to_string(index=False))

print("\n=== REGISTROS EVALUABLES ===")
print(f"Total: {len(evaluables)}")

print("\nCon radicado de respuesta:")
print(
    evaluables["TIENE_RADICADO_RESPUESTA"]
    .value_counts(dropna=False)
    .to_string()
)

print("\nReportes generados correctamente.")