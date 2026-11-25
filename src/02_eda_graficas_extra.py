"""Graficas complementarias EDA PQRS, sin datos sinteticos ni etiquetas SLA."""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw' / 'pqrs_sdg_2022_2026.csv'
OUT = ROOT / 'reports' / 'graficas'
OUT.mkdir(parents=True, exist_ok=True)
if not RAW.is_file():
    raise FileNotFoundError(f'No se encontro el CSV real: {RAW}')
df = pd.read_csv(RAW, sep=';', encoding='utf-8-sig', low_memory=False)
required = {'FECHA_REGISTRO','FECHA_RESPUESTA','TIPO_PETICION','MEDIO_RECEPCION','NOMBRE_LOCALIDAD','SUBTEMA'}
if required - set(df.columns):
    raise ValueError(f'Faltan columnas: {required-set(df.columns)}')
sns.set_theme(style='whitegrid', palette='deep')
for c in ['FECHA_REGISTRO','FECHA_RESPUESTA']:
    df[c+'_DT'] = pd.to_datetime(df[c], format='%d/%m/%Y %H:%M:%S', errors='coerce')
df['DURACION_CALENDARIO'] = (df['FECHA_RESPUESTA_DT']-df['FECHA_REGISTRO_DT']).dt.total_seconds()/86400
df['MES'] = df['FECHA_REGISTRO_DT'].dt.to_period('M')
df['DIA_SEMANA'] = df['FECHA_REGISTRO_DT'].dt.dayofweek
df['HORA'] = df['FECHA_REGISTRO_DT'].dt.hour
valid = df.loc[df['DURACION_CALENDARIO'].between(0,180)].copy()

def save(name):
    plt.tight_layout()
    plt.savefig(OUT/name, dpi=160, bbox_inches='tight')
    plt.close()

# 1 Histograma y curva KDE de duraciones validas para visualizacion
plt.figure(figsize=(11,5))
sns.histplot(data=valid, x='DURACION_CALENDARIO', bins=45, kde=True)
plt.title('Histograma y densidad: duracion observada (0-180 dias)')
plt.xlabel('Dias calendario, no SLA legal')
save('extra_01_histograma_kde.png')

# 2 Boxplot por tipos mas frecuentes
cats = df['TIPO_PETICION'].value_counts().head(7).index
subset = valid.loc[valid['TIPO_PETICION'].isin(cats)]
if not subset.empty:
    plt.figure(figsize=(12,6))
    sns.boxplot(data=subset, x='DURACION_CALENDARIO', y='TIPO_PETICION', showfliers=False)
    plt.title('Boxplot: duracion observada por tipo (sin mostrar atipicos)')
    plt.xlabel('Dias calendario, no SLA legal')
    save('extra_02_boxplot_tipos.png')

# 3 Violin plot, categorias principales
if not subset.empty:
    plt.figure(figsize=(12,6))
    sns.violinplot(data=subset, x='DURACION_CALENDARIO', y='TIPO_PETICION', cut=0, inner='quartile')
    plt.title('Violin plot: distribucion por tipo de peticion')
    plt.xlabel('Dias calendario, no SLA legal')
    save('extra_03_violin_tipos.png')

# 4 Scatter de fecha contra duracion, muestreo reproducible para legibilidad
scatter = valid.dropna(subset=['FECHA_REGISTRO_DT','DURACION_CALENDARIO']).sample(n=min(5000,len(valid)), random_state=42)
if not scatter.empty:
    plt.figure(figsize=(12,5))
    sns.scatterplot(data=scatter, x='FECHA_REGISTRO_DT', y='DURACION_CALENDARIO', alpha=.3, s=12)
    plt.title('Scatter plot: fecha de registro vs duracion (muestra de hasta 5.000)')
    plt.xlabel('Fecha de registro');plt.ylabel('Dias calendario, no SLA legal')
    plt.xticks(rotation=25)
    save('extra_04_scatter_fechas.png')

# 5 Heatmap dias semana x hora
heat = pd.crosstab(df['DIA_SEMANA'], df['HORA']).reindex(index=range(7),columns=range(24),fill_value=0)
heat.index = ['Lun','Mar','Mie','Jue','Vie','Sab','Dom']
plt.figure(figsize=(14,5))
sns.heatmap(heat, cmap='Blues')
plt.title('Heatmap: radicaciones por dia y hora')
plt.xlabel('Hora');plt.ylabel('Dia de la semana')
save('extra_05_heatmap_dia_hora.png')

# 6 Heatmap tipo x medio (top)
top_t = df['TIPO_PETICION'].value_counts().head(8).index
top_m = df['MEDIO_RECEPCION'].value_counts().head(8).index
ct = pd.crosstab(df['TIPO_PETICION'],df['MEDIO_RECEPCION']).reindex(index=top_t,columns=top_m,fill_value=0)
plt.figure(figsize=(12,6))
sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Heatmap: tipo de peticion y medio de recepcion')
save('extra_06_heatmap_tipo_medio.png')

# 7 Countplot de medios
plt.figure(figsize=(12,5))
order=df['MEDIO_RECEPCION'].fillna('SIN DATO').value_counts().head(10).index
sns.countplot(data=df.assign(MEDIO_RECEPCION=df['MEDIO_RECEPCION'].fillna('SIN DATO')), y='MEDIO_RECEPCION', order=order)
plt.title('Countplot: medios de recepcion')
save('extra_07_countplot_medios.png')

# 8 Lineplot por mes
monthly = df.groupby('MES',dropna=True).size()
if not monthly.empty:
    plt.figure(figsize=(13,5))
    sns.lineplot(x=monthly.index.to_timestamp(), y=monthly.values, marker='o')
    plt.title('Lineplot: radicaciones por mes')
    plt.xlabel('Mes');plt.ylabel('Cantidad de PQRS')
    plt.xticks(rotation=25)
    save('extra_08_lineplot_mensual.png')

# 9 Valores faltantes top 15
miss=df.drop(columns=['FECHA_REGISTRO_DT','FECHA_RESPUESTA_DT','DURACION_CALENDARIO','MES','DIA_SEMANA','HORA']).isna().mean().mul(100).sort_values(ascending=False).head(15)
plt.figure(figsize=(11,6))
sns.barplot(x=miss.values,y=miss.index)
plt.title('Porcentaje de valores faltantes por columna')
plt.xlabel('% de registros sin dato')
save('extra_09_faltantes.png')

# 10 Situacion de fechas, sin inferir SLA
state=pd.Series(np.select([df['FECHA_RESPUESTA_DT'].isna(),df['DURACION_CALENDARIO']<0],['Sin fecha de respuesta','Diferencia negativa'],default='Diferencia no negativa')).value_counts()
plt.figure(figsize=(9,4))
sns.barplot(x=state.values,y=state.index)
plt.title('Calidad de fechas de respuesta (no indica cumplimiento SLA)')
plt.xlabel('Registros')
save('extra_10_calidad_fechas.png')
print(f'EDA ampliado finalizado. Graficas guardadas en: {OUT}')
print(f'Registros reales: {len(df):,}; duraciones de 0 a 180 dias usadas para graficas: {len(valid):,}')
