from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'pqrs_sdg_2022_2026.csv'
OUT = ROOT / 'data' / 'processed'
MET = ROOT / 'reports' / 'metricas'
OUT.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
if not RAW.is_file():
    raise FileNotFoundError(f'Falta el CSV REAL: {RAW}')
df = pd.read_csv(RAW, sep=';', encoding='utf-8-sig', low_memory=False)
required = ['FECHA_REGISTRO','FECHA_RESPUESTA','TIPO_PETICION','TIEMPO_TRAMITE','SUBTEMA','MEDIO_RECEPCION','NOMBRE_LOCALIDAD']
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f'Faltan columnas: {missing}')
registro = pd.to_datetime(df['FECHA_REGISTRO'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
respuesta = pd.to_datetime(df['FECHA_RESPUESTA'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
plazo = pd.to_numeric(df['TIEMPO_TRAMITE'], errors='coerce')
duracion = (respuesta - registro).dt.total_seconds() / 86400

# Dataset SOLO con atributos disponibles al momento de radicación.
# No incluir FECHA_RESPUESTA, TIEMPO_TRAMITE, estado actual ni campos de texto libre.
features = pd.DataFrame({
    'SUBTEMA': df['SUBTEMA'].fillna('SIN_DATO').astype(str).str.strip(),
    'TIPO_PETICION': df['TIPO_PETICION'].fillna('SIN_DATO').astype(str).str.strip(),
    'MEDIO_RECEPCION': df['MEDIO_RECEPCION'].fillna('SIN_DATO').astype(str).str.strip(),
    'NOMBRE_LOCALIDAD': df['NOMBRE_LOCALIDAD'].fillna('SIN_DATO').astype(str).str.strip(),
    'DIA_SEMANA': registro.dt.dayofweek.astype('Int64'),
    'HORA_REGISTRO': registro.dt.hour.astype('Int64'),
    'MES': registro.dt.month.astype('Int64'),
    'FECHA_REGISTRO_MODELO': registro.dt.strftime('%Y-%m-%d')
})
features.to_csv(OUT / 'pqrs_variables_iniciales.csv', index=False, encoding='utf-8-sig')

# Tabla agregada para auditar la relación entre tipo de petición y plazo.
plazos = pd.DataFrame({'tipo': df['TIPO_PETICION'].fillna('SIN_DATO'), 'plazo_registrado': plazo})
plazos.groupby(['tipo','plazo_registrado'], dropna=False).size().reset_index(name='registros').sort_values('registros', ascending=False).to_csv(MET / 'diagnostico_plazos_por_tipo.csv', index=False, encoding='utf-8-sig')

# Diagnóstico de calidad SIN exportar identificadores ni datos personales.
condiciones = [
    registro.isna(),
    respuesta.isna(),
    duracion < 0,
    plazo.isna() | (plazo <= 0),
]
nombres = ['FECHA_REGISTRO_INVALIDA','SIN_FECHA_RESPUESTA','RESPUESTA_ANTERIOR_A_REGISTRO','PLAZO_NO_UTILIZABLE']
# Etiquetas no excluyentes: cada condición se cuenta por separado.
resumen = pd.DataFrame({'condicion': nombres, 'registros': [int(c.sum()) for c in condiciones]})
resumen.to_csv(MET / 'diagnostico_calidad_sla.csv', index=False, encoding='utf-8-sig')

# Estado de elegibilidad para revisar manualmente antes de crear el objetivo.
# Un caso con fecha válida NO demuestra que su plazo legal sea correcto.
elegible = registro.notna() & respuesta.notna() & (duracion >= 0) & plazo.notna() & (plazo > 0)
print('\nFASE 4 TERMINADA')
print(f'Filas en variables iniciales: {len(features):,}')
print(f'Casos con fechas ordenadas y plazo registrado positivo: {int(elegible.sum()):,}')
print(resumen.to_string(index=False))
print(f'Dataset: {OUT / "pqrs_variables_iniciales.csv"}')
print('IMPORTANTE: aún NO se generó INCUMPLE_SLA.')
print('Antes del entrenamiento se deben validar plazo aplicable, días hábiles, festivos, suspensiones y significado de FECHA_RESPUESTA.')
