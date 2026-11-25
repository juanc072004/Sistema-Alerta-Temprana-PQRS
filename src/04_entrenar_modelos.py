from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, RocCurveDisplay, precision_recall_curve
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT/'data'/'processed'/'pqrs_etiquetado_validado.csv'
MODELS = ROOT/'models'; MET = ROOT/'reports'/'metricas'; FIG = ROOT/'reports'/'graficas'
for p in [MODELS,MET,FIG]: p.mkdir(parents=True,exist_ok=True)
FEATURES = ['SUBTEMA','TIPO_PETICION','MEDIO_RECEPCION','NOMBRE_LOCALIDAD','DIA_SEMANA','HORA_REGISTRO','MES']
CATEGORICAS = FEATURES[:4]; NUMERICAS = FEATURES[4:]
if not INPUT.is_file():
    raise FileNotFoundError('No existe data/processed/pqrs_etiquetado_validado.csv. FASE 4 NO HA VALIDADO EL SLA. No se permite entrenar con etiquetas inventadas o plazos sin verificar.')
df = pd.read_csv(INPUT,low_memory=False)
required = FEATURES + ['INCUMPLE_SLA','FECHA_REGISTRO_MODELO']
missing = sorted(set(required)-set(df.columns))
if missing: raise ValueError(f'Faltan columnas obligatorias: {missing}')
if not df['INCUMPLE_SLA'].notna().all() or not set(df['INCUMPLE_SLA'].unique()).issubset({0,1}):
    raise ValueError('INCUMPLE_SLA debe contener exclusivamente 0 y 1, sin nulos; solo etiquetas verificadas.')
if df['INCUMPLE_SLA'].nunique()<2: raise ValueError('Se requieren ambas clases para evaluar el modelo.')
df['fecha'] = pd.to_datetime(df['FECHA_REGISTRO_MODELO'],errors='coerce')
if df['fecha'].isna().any(): raise ValueError('Hay fechas de registro inválidas.')
df = df.sort_values('fecha').reset_index(drop=True)
# División temporal: pasado para entrenar, futuro para evaluar.
cut = int(len(df)*0.8)
train,test = df.iloc[:cut].copy(),df.iloc[cut:].copy()
if train['INCUMPLE_SLA'].nunique()<2 or test['INCUMPLE_SLA'].nunique()<2:
    raise ValueError('La partición temporal no contiene ambas clases. Revisar periodos y etiquetas.')
for d in (train,test):
    for col in CATEGORICAS: d[col] = d[col].fillna('SIN_DATO').astype(str)
    for col in NUMERICAS: d[col] = pd.to_numeric(d[col],errors='coerce')
pre = ColumnTransformer([
    ('cat',Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('encoder',OneHotEncoder(handle_unknown='ignore'))]),CATEGORICAS),
    ('num',SimpleImputer(strategy='median'),NUMERICAS)
])
models = {
    'random_forest':RandomForestClassifier(n_estimators=200,min_samples_leaf=3,class_weight='balanced',random_state=42,n_jobs=-1),
    'xgboost':XGBClassifier(n_estimators=250,max_depth=5,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,objective='binary:logistic',eval_metric='logloss',random_state=42,n_jobs=4)
}
rows=[]
for name,est in models.items():
    pipe=Pipeline([('pre',pre),('model',est)])
    pipe.fit(train[FEATURES],train['INCUMPLE_SLA'])
    proba=pipe.predict_proba(test[FEATURES])[:,1]
    pred=(proba>=0.5).astype(int) # umbral inicial de evaluación, NO umbral operativo validado
    report=classification_report(test['INCUMPLE_SLA'],pred,output_dict=True,zero_division=0)
    auc=roc_auc_score(test['INCUMPLE_SLA'],proba)
    rows.append({'modelo':name,'auc_roc':auc,'precision_clase_1':report['1']['precision'],'recall_clase_1':report['1']['recall'],'f1_clase_1':report['1']['f1-score'],'umbral_evaluacion':0.5,'filas_train':len(train),'filas_test':len(test)})
    joblib.dump(pipe,MODELS/f'{name}.joblib')
    (MET/f'{name}_clasificacion.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    ConfusionMatrixDisplay.from_predictions(test['INCUMPLE_SLA'],pred,cmap='Blues',values_format='d');plt.title(f'Matriz de confusión: {name}');plt.tight_layout();plt.savefig(FIG/f'{name}_matriz_confusion.png',dpi=160);plt.close()
    RocCurveDisplay.from_predictions(test['INCUMPLE_SLA'],proba);plt.title(f'ROC: {name}');plt.tight_layout();plt.savefig(FIG/f'{name}_roc.png',dpi=160);plt.close()
    p,r,_=precision_recall_curve(test['INCUMPLE_SLA'],proba);plt.plot(r,p);plt.xlabel('Recall');plt.ylabel('Precisión');plt.title(f'Precision-Recall: {name}');plt.tight_layout();plt.savefig(FIG/f'{name}_precision_recall.png',dpi=160);plt.close()
    print(f'{name}: AUC={auc:.3f}, recall clase 1={report["1"]["recall"]:.3f}')
pd.DataFrame(rows).to_csv(MET/'comparacion_modelos.csv',index=False,encoding='utf-8-sig')
(MODELS/'metadata.json').write_text(json.dumps({'features':FEATURES,'train_max_date':str(train['fecha'].max().date()),'test_min_date':str(test['fecha'].min().date()),'advertencia':'Los umbrales de alerta deben calibrarse y validarse antes de uso operativo.'},ensure_ascii=False,indent=2),encoding='utf-8')
print('Entrenamiento finalizado. Revisa reports/metricas/comparacion_modelos.csv')
