from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'pqrs_sdg_2022_2026.csv'
GRAF = ROOT / 'reports' / 'graficas'
MET = ROOT / 'reports' / 'metricas'
GRAF.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)
if not RAW.is_file():
    raise FileNotFoundError(f'Falta el CSV REAL: {RAW}')
df = pd.read_csv(RAW, sep=';', encoding='utf-8-sig', low_memory=False)
required = ['FECHA_REGISTRO', 'FECHA_RESPUESTA', 'TIPO_PETICION', 'TIEMPO_TRAMITE', 'SUBTEMA', 'MEDIO_RECEPCION', 'NOMBRE_LOCALIDAD']
missing = sorted(set(required) - set(df.columns))
if missing:
    raise ValueError(f'Columnas requeridas ausentes: {missing}')
for c in ['FECHA_REGISTRO', 'FECHA_RESPUESTA', 'FECHA_ASIGNACION', 'FECHA_SEGUIMIENTO']:
    if c in df:
        df[c + '_DT'] = pd.to_datetime(df[c], format='%d/%m/%Y %H:%M:%S', errors='coerce')
registro = df['FECHA_REGISTRO_DT']
respuesta = df['FECHA_RESPUESTA_DT']
dias = (respuesta - registro).dt.total_seconds() / 86400

summary = pd.DataFrame([{
    'registros': len(df), 'columnas_originales': len(pd.read_csv(RAW, sep=';', encoding='utf-8-sig', nrows=0).columns),
    'duplicados_exactos': int(df[[c for c in df if not c.endswith('_DT')]].duplicated().sum()),
    'sin_fecha_respuesta': int(respuesta.isna().sum()),
    'diferencia_negativa_respuesta_menos_registro': int((dias < 0).sum()),
    'sin_tipo_peticion': int(df['TIPO_PETICION'].isna().sum()),
    'sin_tiempo_tramite': int(df['TIEMPO_TRAMITE'].isna().sum())
}])
summary.to_csv(MET / 'eda_resumen.csv', index=False, encoding='utf-8-sig')

for col, fname in [('TIPO_PETICION','eda_tipos.csv'), ('MEDIO_RECEPCION','eda_medios.csv'), ('NOMBRE_LOCALIDAD','eda_localidades.csv'), ('SUBTEMA','eda_subtemas.csv'), ('TIEMPO_TRAMITE','eda_plazos_registrados.csv')]:
    df[col].fillna('SIN DATO').astype(str).value_counts().rename_axis(col).reset_index(name='cantidad').to_csv(MET / fname, index=False, encoding='utf-8-sig')

by_month = registro.dt.to_period('M').value_counts().sort_index()
by_month.rename_axis('mes').reset_index(name='cantidad').assign(mes=lambda x: x['mes'].astype(str)).to_csv(MET / 'eda_registros_por_mes.csv', index=False, encoding='utf-8-sig')

def bar_top(col, title, fname, n=10):
    counts = df[col].fillna('SIN DATO').astype(str).value_counts().head(n).sort_values()
    fig, ax = plt.subplots(figsize=(12, 6))
    counts.plot(kind='barh', ax=ax)
    ax.set_title(title)
    ax.set_xlabel('Número de registros')
    fig.tight_layout()
    fig.savefig(GRAF / fname, dpi=150)
    plt.close(fig)

bar_top('TIPO_PETICION', 'Tipos de petición más frecuentes', 'eda_tipos.png')
bar_top('NOMBRE_LOCALIDAD', 'Localidades más frecuentes', 'eda_localidades.png')
bar_top('MEDIO_RECEPCION', 'Medios de recepción', 'eda_medios.png')
bar_top('SUBTEMA', 'Subtemas más frecuentes', 'eda_subtemas.png')

fig, ax = plt.subplots(figsize=(12, 5))
by_month.plot(ax=ax)
ax.set_title('Radicaciones por mes')
ax.set_xlabel('Mes de registro')
ax.set_ylabel('Número de registros')
fig.tight_layout()
fig.savefig(GRAF / 'eda_registros_por_mes.png', dpi=150)
plt.close(fig)

# Las duraciones negativas se conservan y se informan; no se corrigen automáticamente.
positive = dias[(dias >= 0) & (dias <= 180)].dropna()
if not positive.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(positive, bins=40)
    ax.set_title('Duración observada de respuesta (0 a 180 días calendario)')
    ax.set_xlabel('Días calendario transcurridos')
    ax.set_ylabel('Registros')
    fig.tight_layout()
    fig.savefig(GRAF / 'eda_duracion_respuesta.png', dpi=150)
    plt.close(fig)

print('\nFASE 3 TERMINADA')
print(summary.to_string(index=False))
print(f'Gráficas: {GRAF}')
print(f'Reportes: {MET}')
print('ATENCIÓN: duración calendario observada NO equivale a incumplimiento legal de SLA.')
