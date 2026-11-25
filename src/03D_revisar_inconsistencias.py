
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

for columna in ["RADI_FECH_RADI", "FECHA_RESPUESTA"]:
    df[columna] = pd.to_datetime(
        df[columna],
        errors="coerce",
        dayfirst=True
    )

df["DIAS_RESPUESTA"] = (
    df["FECHA_RESPUESTA"] - df["RADI_FECH_RADI"]
).dt.days

# Casos con respuesta anterior a la radicación
negativos = df[df["DIAS_RESPUESTA"] < 0].copy()

# Casos sin respuesta registrada
sin_respuesta = df[
    df["FECHA_RESPUESTA"].isna()
].copy()

# Casos con cronología no negativa
cronologia_valida = df[
    df["DIAS_RESPUESTA"].notna()
    & (df["DIAS_RESPUESTA"] >= 0)
].copy()

# Exportar muestras para revisión
negativos.head(100).to_csv(
    SALIDA / "muestra_fechas_negativas.csv",
    index=False,
    encoding="utf-8-sig"
)

sin_respuesta.head(100).to_csv(
    SALIDA / "muestra_sin_respuesta.csv",
    index=False,
    encoding="utf-8-sig"
)

resumen = pd.DataFrame([{
    "total_registros": len(df),
    "cronologia_no_negativa": len(cronologia_valida),
    "fechas_negativas": len(negativos),
    "sin_fecha_respuesta": len(sin_respuesta)
}])

resumen.to_csv(
    SALIDA / "resumen_inconsistencias.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== REVISION DE INCONSISTENCIAS ===")
print(f"Total: {len(df)}")
print(f"Cronologia no negativa: {len(cronologia_valida)}")
print(f"Fechas negativas: {len(negativos)}")
print(f"Sin fecha de respuesta: {len(sin_respuesta)}")
print("\nArchivos generados correctamente.")