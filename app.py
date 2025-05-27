
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import requests

st.set_page_config(page_title="Simulador de Subsidios - Análisis BALI", layout="wide")
st.title("📊 Simulador de Subsidios con Análisis BALI + Chat")

API_KEY = st.secrets["CHATPDF_API_KEY"]
SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"

def predecir_monto(df):
    modelo = LinearRegression()
    X = df[['Año']]
    y = df['Monto']
    modelo.fit(X, y)
    prediccion = modelo.predict([[2025]])[0]
    return prediccion

def graficar(df, subsidio):
    pred = predecir_monto(df)
    df_plot = df.copy()
    df_plot.loc[len(df_plot.index)] = [2025, pred]
    fig = px.line(df_plot, x='Año', y='Monto', markers=True, title=f"{subsidio} - Histórico y Proyección")
    fig.add_scatter(x=[2025], y=[pred], mode='markers+text', text=[f"Proy: ${pred:,.0f}"], marker=dict(size=12, color='red'))
    st.plotly_chart(fig, use_container_width=True)
    return pred

def analizar_tendencia(df, subsidio):
    crecimiento = (df['Monto'].iloc[-1] - df['Monto'].iloc[0]) / df['Monto'].iloc[0]
    if crecimiento > 0.1:
        st.success(f"🔼 {subsidio}: Crecimiento positivo. Revisar cumplimiento de metas BALI.")
    elif crecimiento < -0.1:
        st.error(f"🔽 {subsidio}: Caída significativa. Puede indicar incumplimientos contractuales.")
    else:
        st.info(f"➡️ {subsidio}: Comportamiento estable. Seguir monitoreando indicadores según el BALI.")
    return crecimiento

def consultar_bali(texto, subsidio):
    st.markdown(f"**Consulta automática sobre el contrato respecto a {subsidio}...**")
    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    data = {
        "sourceId": SOURCE_ID,
        "messages": [{ "role": "user", "content": texto }]
    }
    response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)
    if response.status_code == 200:
        result = response.json()
        st.markdown(result["content"])
    else:
        st.error("Error al consultar el contrato BALI. Verifica tu API Key o Source ID.")

subsidios = {
    "Subsidio Variable": [2021, 2022, 2023, 2024], [816_375_829, 2_316_612_803, 1_963_167_525, 2_319_599_141],
    "Subsidio Fijo": [2021, 2022, 2023, 2024], [1_000_000_000, 1_020_000_000, 1_040_000_000, 1_060_000_000],
    "Subsidio Complementario": [2021, 2022, 2023, 2024], [300_000_000, 320_000_000, 280_000_000, 350_000_000],
    "Subsidio Especial": [2021, 2022, 2023, 2024], [150_000_000, 130_000_000, 180_000_000, 160_000_000],
}

for subsidio, (anios, montos) in subsidios.items():
    with st.expander(f"📁 {subsidio}"):
        st.subheader(f"{subsidio} - Datos históricos")
        df = pd.DataFrame({'Año': anios, 'Monto': montos})
        df_edit = st.data_editor(df, use_container_width=True, key=subsidio)
        pred = graficar(df_edit, subsidio)
        analizar_tendencia(df_edit, subsidio)
        consultar_bali(f"Analiza el comportamiento del {subsidio} según el contrato BALI y entrega observaciones si supera o cae bajo lo esperado.", subsidio)

st.divider()
st.subheader("💬 Consulta manual al contrato BALI")
user_input = st.chat_input("Escribe tu pregunta sobre el contrato...")
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.spinner("🔎 Consultando el contrato..."):
        consultar_bali(user_input, "Consulta manual")
