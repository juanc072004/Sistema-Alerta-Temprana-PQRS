
from pathlib import Path
import pandas as pd

# ============================================
# 1. CONFIGURACION DE RUTAS
# ============================================

BASE_DIR = Path(__file__).resolve().parent.parent

RUTA_CSV = BASE_DIR / "data" / "raw" / "pqrs_sdg_2022_2026.csv"
RUTA_REPORTES = BASE_DIR / "reports" / "metricas"

RUTA_REPORTES.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. CARGA EXCLUSIVA DE DATOS REALES
# ============================================

if not RUTA_CSV.exists():
    raise FileNotFoundError(
        f"No se encontro el dataset real: {RUTA_CSV}\n"
        "Coloca el CSV original en data/raw/. "
        "No se generaran datos simulados."
    )

print("Cargando dataset real del SDQS...")

df = pd.read_csv(
    RUTA_CSV,
    sep=";",
    encoding="utf-8-sig",
    low_memory=False
)

print("\n========== DIMENSIONES ==========")
print(f"Filas: {df.shape[0]:,}")
print(f"Columnas: {df.shape[1]}")

print("\n========== COLUMNAS ==========")
for columna in df.columns:
    print("-", columna)

# ============================================
# 3. TIPOS DE DATOS
# ============================================

print("\n========== TIPOS DE DATOS ==========")
print(df.dtypes)

# ============================================
# 4. VALORES FALTANTES
# ============================================

faltantes = pd.DataFrame({
    "columna": df.columns,
    "faltantes": df.isna().sum().values,
    "porcentaje": (
        df.isna().mean().values * 100
    ).round(2)
})

faltantes = faltantes.sort_values(
    by="faltantes",
    ascending=False
)

print("\n========== VALORES FALTANTES ==========")
print(faltantes.to_string(index=False))

faltantes.to_csv(
    RUTA_REPORTES / "valores_faltantes.csv",
    index=False,
    encoding="utf-8-sig"
)

# ============================================
# 5. DUPLICADOS
# ============================================

duplicados = df.duplicated().sum()

print("\n========== DUPLICADOS ==========")
print(f"Registros duplicados exactos: {duplicados:,}")

# ============================================
# 6. REVISION DE FECHAS
# ============================================

columnas_fecha = [
    "FECHA_REGISTRO",
    "FECHA_RESPUESTA",
    "FECHA_ASIGNACION",
    "FECHA_SEGUIMIENTO"
]

for columna in columnas_fecha:
    if columna in df.columns:
        df[columna] = pd.to_datetime(
            df[columna],
            errors="coerce",
            dayfirst=True
        )

print("\n========== RESUMEN DE FECHAS ==========")

for columna in columnas_fecha:
    if columna in df.columns:
        print(f"\n{columna}")
        print("Fechas validas:", df[columna].notna().sum())
        print("Fecha minima:", df[columna].min())
        print("Fecha maxima:", df[columna].max())

# ============================================
# 7. DIFERENCIA ENTRE FECHAS
# ============================================

if {
    "FECHA_REGISTRO",
    "FECHA_RESPUESTA"
}.issubset(df.columns):

    df["DIAS_CALENDARIO"] = (
        df["FECHA_RESPUESTA"] -
        df["FECHA_REGISTRO"]
    ).dt.total_seconds() / 86400

    print("\n========== REVISION TEMPORAL ==========")

    print(
        "Sin fecha de respuesta:",
        df["FECHA_RESPUESTA"].isna().sum()
    )

    print(
        "Fechas de respuesta anteriores al registro:",
        (df["DIAS_CALENDARIO"] < 0).sum()
    )

    print(
        "Duraciones iguales a cero:",
        (df["DIAS_CALENDARIO"] == 0).sum()
    )

    print(
        "Duraciones positivas:",
        (df["DIAS_CALENDARIO"] > 0).sum()
    )

    print("\nEstadisticas de dias calendario:")
    print(df["DIAS_CALENDARIO"].describe())

# ============================================
# 8. CATEGORIAS PRINCIPALES
# ============================================

columnas_categoricas = [
    "SUBTEMA",
    "TIPO_PETICION",
    "MEDIO_RECEPCION",
    "NOMBRE_LOCALIDAD",
    "ESTADO_ORFEO"
]

for columna in columnas_categoricas:
    if columna in df.columns:
        print(f"\n========== {columna} ==========")
        print(df[columna].value_counts(dropna=False).head(15))

# ============================================
# 9. EXPORTAR INFORME DE AUDITORIA
# ============================================

resumen = {
    "total_registros": len(df),
    "total_columnas": len(df.columns),
    "duplicados_exactos": int(duplicados),
    "sin_fecha_respuesta": int(
        df["FECHA_RESPUESTA"].isna().sum()
    ) if "FECHA_RESPUESTA" in df.columns else None,
    "fechas_negativas": int(
        (df["DIAS_CALENDARIO"] < 0).sum()
    ) if "DIAS_CALENDARIO" in df.columns else None
}

pd.DataFrame([resumen]).to_csv(
    RUTA_REPORTES / "resumen_auditoria.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n===================================")
print("AUDITORIA FINALIZADA")
print("Los datos originales no fueron modificados.")
print("Informes guardados en reports/metricas/")
print("===================================")