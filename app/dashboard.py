from pathlib import Path
import json
import pandas as pd
import streamlit as st
import requests
import joblib

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'data'/'raw'/'pqrs_sdg_2022_2026.csv'
MODEL=ROOT/'models'/'random_forest.joblib'
MET=ROOT/'reports'/'metricas'/'comparacion_modelos.csv'
FEATURES=['SUBTEMA','TIPO_PETICION','MEDIO_RECEPCION','NOMBRE_LOCALIDAD','DIA_SEMANA','HORA_REGISTRO','MES']
st.set_page_config(page_title='PQRS | Alerta temprana',page_icon='📊',layout='wide')
st.title('Sistema de Alerta Temprana — PQRS')
st.caption('Análisis de datos reales. Las predicciones son apoyo preventivo, no determinaciones legales.')
if not RAW.is_file(): st.error('No se encontró el CSV real en data/raw.');st.stop()
@st.cache_data
def cargar():
    df=pd.read_csv(RAW,sep=';',encoding='utf-8-sig',low_memory=False)
    df['fecha']=pd.to_datetime(df['FECHA_REGISTRO'],format='%d/%m/%Y %H:%M:%S',errors='coerce')
    return df
df=cargar()
a,b,c=st.columns(3)
a.metric('Peticiones',f'{len(df):,}')
b.metric('Sin fecha de respuesta',f'{df["FECHA_RESPUESTA"].isna().sum():,}')
c.metric('Tipos de petición',f'{df["TIPO_PETICION"].nunique(dropna=True):,}')
t1,t2,t3=st.tabs(['Exploración','Predicción','Asistente local'])
with t1:
    st.subheader('Radicaciones mensuales')
    serie=df.dropna(subset=['fecha']).set_index('fecha').resample('MS').size()
    st.line_chart(serie)
    st.subheader('Tipos de petición más frecuentes')
    st.bar_chart(df['TIPO_PETICION'].fillna('SIN_DATO').value_counts().head(12))
    if MET.is_file():
        st.subheader('Comparación de modelos')
        st.dataframe(pd.read_csv(MET),hide_index=True)
with t2:
    if not MODEL.is_file():
        st.warning('Todavía no hay modelo entrenado. Debes validar INCUMPLE_SLA antes de ejecutar la fase 5.')
    else:
        st.warning('Modelo experimental: probabilidades sin calibración operativa ni validación jurídica del SLA.')
        with st.form('form_pred'):
            values={}
            for col in FEATURES[:4]:
                opts=sorted(df[col].dropna().astype(str).unique().tolist())
                values[col]=st.selectbox(col,opts if opts else ['SIN_DATO'])
            values['DIA_SEMANA']=st.selectbox('Día de la semana (lunes=0)',list(range(7)))
            values['HORA_REGISTRO']=st.slider('Hora de registro',0,23,12)
            values['MES']=st.selectbox('Mes',list(range(1,13)))
            submitted=st.form_submit_button('Calcular riesgo experimental')
        if submitted:
            model=joblib.load(MODEL)
            prob=float(model.predict_proba(pd.DataFrame([values]))[0,1])
            st.metric('Probabilidad estimada de incumplimiento',f'{prob:.1%}')
            st.info('No se asigna semáforo definitivo: faltan umbrales validados y calibración.')
with t3:
    st.subheader('Explicación con Ollama (CPU/local)')
    st.caption('Solo se envía contexto estadístico general. No se envían nombres, identificadores ni textos de peticiones.')
    model_name=st.text_input('Modelo local instalado',value='qwen2.5:1.5b')
    pregunta=st.text_area('Pregunta',value='¿Qué aspectos debería revisar un analista antes de interpretar estos datos?')
    if st.button('Consultar Ollama'):
        contexto=f'Total de registros: {len(df)}. Fechas de respuesta faltantes: {int(df["FECHA_RESPUESTA"].isna().sum())}. El SLA no está validado.'
        prompt='Eres un asistente técnico de PQRS. No inventes datos ni concluyas incumplimientos legales. Responde de forma breve en español.\n'+contexto+'\nPregunta: '+pregunta
        try:
            r=requests.post('http://localhost:11434/api/generate',json={'model':model_name,'prompt':prompt,'stream':False},timeout=180)
            r.raise_for_status()
            st.write(r.json().get('response','Sin respuesta del modelo.'))
        except requests.RequestException as exc:
            st.error(f'No se pudo conectar con Ollama: {exc}. Comprueba que está iniciado y que el modelo existe.')
