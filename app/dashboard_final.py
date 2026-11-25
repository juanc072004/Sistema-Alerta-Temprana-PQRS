from pathlib import Path
import json
import pandas as pd
import streamlit as st
import joblib
import requests

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/metricas'
MODEL_DIR = ROOT / 'models'
st.set_page_config(page_title='Alerta Temprana PQRS',page_icon='📊',layout='wide')
st.title('Sistema de Alerta Temprana · PQRS')
st.caption('Proyecto académico | Riesgo de superar el plazo nominal de respuesta')
st.warning('Las alertas son experimentales. No representan incumplimientos legales confirmados.')

meta_path = REPORTS/'metadata_final.json'
if not meta_path.exists():
    st.error('Primero ejecuta: python src/05_entrenar_version_final.py')
    st.stop()
meta = json.loads(meta_path.read_text(encoding='utf-8'))
model_path = MODEL_DIR/f"{meta['modelo_seleccionado']}_final.joblib"
if not model_path.exists():
    st.error(f'No existe el modelo {model_path.name}')
    st.stop()

@st.cache_resource
def load_model(path):
    return joblib.load(path)
model = load_model(str(model_path))

page1,page2,page3,page4 = st.tabs(['Indicadores y resultados','Simular petición','Análisis de prueba','Asistente local'])
with page1:
    a,b,c,d = st.columns(4)
    a.metric('Modelo seleccionado',meta['modelo_seleccionado'].replace('_',' ').title())
    b.metric('Entrenamiento',f"{meta['registros_train']:,}")
    c.metric('Validación',f"{meta['registros_val']:,}")
    d.metric('Prueba',f"{meta['registros_test']:,}")
    comparison = REPORTS/'comparacion_final.csv'
    if comparison.exists():
        table = pd.read_csv(comparison)
        st.subheader('Comparación fuera de muestra')
        st.dataframe(table,hide_index=True,use_container_width=True)
        st.bar_chart(table.set_index('modelo')[['test_pr_auc','test_roc_auc']])
    st.info(f"Umbral seleccionado en validación: {meta['umbral_alerta']:.3f}. La prueba final no se usó para elegirlo.")
with page2:
    st.subheader('Predicción al momento de la radicación')
    st.caption('Usar únicamente atributos disponibles al radicar. No introducir datos personales.')
    with st.form('prediccion'):
        tipo = st.selectbox('Tipo de petición',meta['categorias']['TIPO_PETICION'])
        medio = st.selectbox('Medio de recepción',meta['categorias']['MEDIO_RECEPCION'])
        localidad = st.selectbox('Localidad',meta['categorias']['NOMBRE_LOCALIDAD'])
        fecha = st.date_input('Fecha de radicación')
        send = st.form_submit_button('Calcular riesgo experimental')
    if send:
        row = pd.DataFrame([{'TIPO_PETICION':tipo,'MEDIO_RECEPCION':medio,
                             'NOMBRE_LOCALIDAD':localidad,'ANIO_RADICACION':fecha.year,
                             'MES_RADICACION':fecha.month,'DIA_SEMANA_RADICACION':fecha.weekday()}])
        p = float(model.predict_proba(row)[0,1])
        st.metric('Puntuación del modelo',f'{p:.1%}')
        if p >= meta['umbral_alerta']:
            st.warning('ALERTA EXPERIMENTAL: conviene priorizar seguimiento humano.')
        else:
            st.success('Sin alerta bajo el umbral seleccionado; no garantiza cumplimiento.')
        st.caption('La puntuación no es una probabilidad calibrada ni un dictamen jurídico.')
with page3:
    path = REPORTS/'predicciones_prueba_final.csv'
    if path.exists():
        test = pd.read_csv(path,parse_dates=['RADI_FECH_RADI'])
        st.subheader('Resultados del periodo de prueba')
        st.line_chart(test.set_index('RADI_FECH_RADI').resample('MS')['y'].agg(['sum','count']))
        st.caption('Suma mensual de casos que superaron el plazo nominal y volumen total de registros.')
        st.dataframe(test.groupby('TIPO_PETICION').agg(
            peticiones=('y','size'),superaciones=('y','sum'),
            puntuacion_media=('probabilidad_modelo','mean')
        ).sort_values('peticiones',ascending=False),use_container_width=True)
with page4:
    st.subheader('Asistente con Ollama (local)')
    st.caption('Explica resultados agregados. No se envían radicados, observaciones ni datos personales.')
    model_name = st.text_input('Modelo local de Ollama','qwen2.5:1.5b')
    question = st.text_area('Pregunta sobre el experimento','¿Cómo debo interpretar el recall y las falsas alarmas?')
    if st.button('Consultar Ollama'):
        if not question.strip():
            st.error('Escribe una pregunta.')
        else:
            try:
                comp = pd.read_csv(REPORTS/'comparacion_final.csv')
                context = comp[['modelo','test_pr_auc','test_precision','test_recall','test_fp','test_fn']].round(4).to_json(orient='records')
                instruction = ('Eres un asistente académico. Responde en español, con claridad. '
                    'Solo dispones de métricas agregadas de un experimento de riesgo de superar plazos nominales, '
                    'NO de incumplimiento legal probado. No inventes resultados ni cites normativa no verificada. '
                    f'Métricas: {context}\nPregunta: {question[:1200]}')
                with st.spinner('Consultando modelo local...'):
                    r = requests.post('http://127.0.0.1:11434/api/generate',
                        json={'model':model_name,'prompt':instruction,'stream':False},timeout=180)
                    r.raise_for_status()
                    st.write(r.json().get('response','Sin respuesta del modelo.'))
            except requests.RequestException as e:
                st.error(f'No fue posible conectar con Ollama local. Verifica que esté ejecutándose. Detalle: {e}')
            except Exception as e:
                st.error(f'No se pudo completar la consulta: {e}')
