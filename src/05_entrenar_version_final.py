from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, f1_score, fbeta_score, confusion_matrix)
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/processed/pqrs_sla_preliminar.csv'
OUT = ROOT / 'reports/metricas'
MODEL_DIR = ROOT / 'models'
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
if not SOURCE.exists():
    raise FileNotFoundError(f'No se encontró {SOURCE}. Ejecuta primero 03E_calcular_sla.py')

df = pd.read_csv(SOURCE, low_memory=False, encoding='utf-8-sig')
needed = ['ESTADO_EVALUACION','ESTADO_ORFEO','SUPERA_PLAZO_PRELIMINAR','RADI_FECH_RADI',
          'TIPO_PETICION','MEDIO_RECEPCION','NOMBRE_LOCALIDAD']
missing = [c for c in needed if c not in df.columns]
if missing:
    raise ValueError(f'Columnas faltantes: {missing}')

df['RADI_FECH_RADI'] = pd.to_datetime(df['RADI_FECH_RADI'], errors='coerce', dayfirst=False)
closed = df['ESTADO_ORFEO'].astype('string').str.strip().str.upper().eq('TRAMITE CERRADO').fillna(False)
valid = (df['ESTADO_EVALUACION'].eq('EVALUACION_PRELIMINAR') & closed &
         df['SUPERA_PLAZO_PRELIMINAR'].notna() & df['RADI_FECH_RADI'].notna())
df = df.loc[valid].copy()
df['y'] = pd.to_numeric(df['SUPERA_PLAZO_PRELIMINAR'], errors='coerce').astype(int)
# Solo variables razonablemente disponibles al radicar. Verificar disponibilidad histórica antes de producción.
cat_cols = ['TIPO_PETICION', 'MEDIO_RECEPCION', 'NOMBRE_LOCALIDAD']
num_cols = ['ANIO_RADICACION', 'MES_RADICACION', 'DIA_SEMANA_RADICACION']
df['ANIO_RADICACION'] = df['RADI_FECH_RADI'].dt.year
df['MES_RADICACION'] = df['RADI_FECH_RADI'].dt.month
df['DIA_SEMANA_RADICACION'] = df['RADI_FECH_RADI'].dt.dayofweek
for col in cat_cols:
    df[col] = df[col].astype('string').fillna('SIN_DATO').str.strip().replace('', 'SIN_DATO').astype(str)
df = df.sort_values('RADI_FECH_RADI').reset_index(drop=True)
unique_days = np.sort(df['RADI_FECH_RADI'].dt.normalize().unique())
if len(unique_days) < 15:
    raise ValueError('No hay suficientes fechas distintas para separar entrenamiento, validación y prueba.')
cut1 = unique_days[int(len(unique_days)*0.60)]
cut2 = unique_days[int(len(unique_days)*0.80)]
day = df['RADI_FECH_RADI'].dt.normalize()
sets = {'train': df.loc[day < cut1].copy(),
        'val': df.loc[(day >= cut1) & (day < cut2)].copy(),
        'test': df.loc[day >= cut2].copy()}
for name, subset in sets.items():
    if len(subset) == 0 or subset['y'].nunique() != 2:
        raise ValueError(f'{name}: no hay ambas clases o está vacío. Revisar periodos.')
    print(f'{name}: {len(subset):,} filas; positivos: {subset.y.sum():,}; '
          f'periodo {subset.RADI_FECH_RADI.min().date()} a {subset.RADI_FECH_RADI.max().date()}')

X_train, y_train = sets['train'][cat_cols+num_cols], sets['train']['y']
X_val, y_val = sets['val'][cat_cols+num_cols], sets['val']['y']
X_test, y_test = sets['test'][cat_cols+num_cols], sets['test']['y']


def preprocessor():
    return ColumnTransformer([
        ('categoricas', Pipeline([
            ('imputar', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
        ('numericas', SimpleImputer(strategy='median'), num_cols)
    ])

ratio = (len(y_train)-int(y_train.sum())) / max(1,int(y_train.sum()))
configs = {
    'random_forest': RandomForestClassifier(n_estimators=240, max_depth=12,
        min_samples_leaf=10, class_weight='balanced_subsample', random_state=42, n_jobs=-1),
    'xgboost': XGBClassifier(n_estimators=350, max_depth=3, learning_rate=0.04,
        subsample=0.85, colsample_bytree=0.9, scale_pos_weight=ratio,
        objective='binary:logistic', eval_metric='logloss', tree_method='hist',
        random_state=42, n_jobs=4)
}

def metrics(y, p, threshold):
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0,1]).ravel()
    return {'pr_auc': float(average_precision_score(y,p)),
            'roc_auc': float(roc_auc_score(y,p)),
            'precision': float(precision_score(y,pred,zero_division=0)),
            'recall': float(recall_score(y,pred,zero_division=0)),
            'f1': float(f1_score(y,pred,zero_division=0)),
            'f2': float(fbeta_score(y,pred,beta=2,zero_division=0)),
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}

# El umbral se elige SOLO con validación. Restricción mínima de precisión
# para evitar elegir un sistema que alerte prácticamente todo.
def choose_threshold(y, probabilities):
    thresholds = np.unique(np.quantile(probabilities, np.linspace(0.05,0.99,100)))
    candidates = []
    for t in thresholds:
        m = metrics(y,probabilities,float(t))
        candidates.append((float(t),m))
    minimum_precision = max(0.08, float(y.mean())*1.5)
    eligible = [(t,m) for t,m in candidates if m['precision'] >= minimum_precision]
    if not eligible:
        warnings.warn('Ningún umbral cumple precisión mínima: se selecciona mayor F2 en validación.')
        eligible = candidates
    return max(eligible,key=lambda pair: (pair[1]['f2'],pair[1]['precision']))[0]

runs = []
for name, clf in configs.items():
    print(f'\nEntrenando {name}...')
    pipeline = Pipeline([('preprocesador',preprocessor()),('modelo',clf)])
    pipeline.fit(X_train,y_train)
    p_val = pipeline.predict_proba(X_val)[:,1]
    threshold = choose_threshold(y_val,p_val)
    vm = metrics(y_val,p_val,threshold)
    print(f'Validación: PR-AUC={vm["pr_auc"]:.4f}; umbral={threshold:.3f}; '
          f'precision={vm["precision"]:.3f}; recall={vm["recall"]:.3f}')
    runs.append((name,pipeline,threshold,vm))

# Elegir modelo sin mirar las métricas de prueba.
winner = max(runs,key=lambda run: run[3]['pr_auc'])[0]
rows = []
for name,pipeline,threshold,vm in runs:
    p_test = pipeline.predict_proba(X_test)[:,1]
    tm = metrics(y_test,p_test,threshold)
    row = {'modelo':name,'umbral_validacion':threshold,'val_pr_auc':vm['pr_auc'],
           'val_precision':vm['precision'],'val_recall':vm['recall'],
           'test_positivos':int(y_test.sum()),'test_total':len(y_test),
           **{f'test_{k}':v for k,v in tm.items()}}
    rows.append(row)
    print(f'Prueba {name}: PR-AUC={tm["pr_auc"]:.4f}; '
          f'precision={tm["precision"]:.3f}; recall={tm["recall"]:.3f}; '
          f'FP={tm["fp"]}; FN={tm["fn"]}')
    joblib.dump(pipeline,MODEL_DIR/f'{name}_final.joblib')
    if name == winner:
        predictions = sets['test'][['RADI_FECH_RADI','TIPO_PETICION','y']].copy()
        predictions['probabilidad_modelo'] = p_test
        predictions['alerta'] = (p_test >= threshold).astype(int)
        predictions.to_csv(OUT/'predicciones_prueba_final.csv',index=False,encoding='utf-8-sig')

pd.DataFrame(rows).to_csv(OUT/'comparacion_final.csv',index=False,encoding='utf-8-sig')
selected = next(r for r in runs if r[0] == winner)
meta = {
    'modelo_seleccionado':winner,'umbral_alerta':selected[2],
    'variables_categoricas':cat_cols,'variables_numericas':num_cols,
    'categorias':{c:sorted(df[c].dropna().astype(str).unique().tolist()) for c in cat_cols},
    'corte_entrenamiento_validacion':str(pd.Timestamp(cut1).date()),
    'corte_validacion_prueba':str(pd.Timestamp(cut2).date()),
    'registros_train':len(X_train),'registros_val':len(X_val),'registros_test':len(X_test),
    'tasa_positivos_prueba':float(y_test.mean()),
    'objetivo':'Superación preliminar de plazo nominal (no incumplimiento legal validado)',
    'advertencia':'Modelo experimental. No utilizar para decisiones sancionatorias ni como certificación legal.'
}
(OUT/'metadata_final.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
print(f'\nMODELO SELECCIONADO POR PR-AUC DE VALIDACION: {winner}')
print('Archivos finales guardados en models/ y reports/metricas/.')
