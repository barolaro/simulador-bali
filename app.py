import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import requests

st.set_page_config(page_title="Simulador Subsidios BALI + Chat", layout="wide")
st.title("📊 Simulador de Subsidios Hospitalarios con Análisis BALI + Chat")

st.markdown("Edita los datos históricos, visualiza la proyección 2025, interpreta automáticamente el comportamiento y haz consultas al contrato BALI.")

API_KEY = st.secrets["CHATPDF_API_KEY"]
SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"  # <- ya confirmado que está activo

def proyeccion_y_comentario(nombre_subsidio, valores):
    df = pd.DataFrame({
        "Año": [2021, 2022, 2023, 2024],
        "Monto": valores
    })
    st.data_editor(df, num_rows="fixed", use_container_width=True)

    modelo = LinearRegression().fit(df[["Año"]], df["Monto"])
    pred = modelo.predict([[2025]])[0]

    fig = px.line(df, x="Año", y="Monto", markers=True, title=f"{nombre_subsidio} - Histórico y Proyección")
    fig.add_scatter(x=[2025], y=[pred], mode='markers+text',
                    text=[f"Proy: ${pred:,.0f}"], textposition='top right',
                    marker=dict(size=10, color='red'), name="Proyección 2025")
    st.plotly_chart(fig, use_container_width=True)

    crecimiento = (valores[-1] - valores[0]) / valores[0]
    if crecimiento > 0.2:
        st.success(f"🔼 {nombre_subsidio} en fuerte alza. Evaluar impacto en resultados financieros y cumplimiento.")
    elif crecimiento < -0.1:
        st.error(f"🔽 {nombre_subsidio} en caída. Requiere revisión conforme al contrato BALI.")
    else:
        st.info(f"➡️ Comportamiento estable. Seguir monitoreando indicadores según el BALI.")

# Sección por subsidio
tabs = st.tabs(["Subsidio Fijo", "Subsidio Variable", "Subsidio Complementario", "Subsidio Especial", "🤖 ChatBali"])

with tabs[0]:
    st.subheader("Subsidio Fijo")
    proyeccion_y_comentario("Subsidio Fijo", [1000000000, 1020000000, 1040000000, 1060000000])

with tabs[1]:
    st.subheader("Subsidio Variable")
    proyeccion_y_comentario("Subsidio Variable", [816375829, 2316612803, 1963167525, 2319599141])

with tabs[2]:
    st.subheader("Subsidio Complementario")
    proyeccion_y_comentario("Subsidio Complementario", [600000000, 630000000, 615000000, 640000000])

with tabs[3]:
    st.subheader("Subsidio Especial")
    proyeccion_y_comentario("Subsidio Especial", [120000000, 130000000, 125000000, 128000000])

with tabs[4]:
    st.subheader("💬 Consultas al Contrato BALI")
    st.markdown("Pregunta lo que necesites sobre el contrato. Se usará el documento PDF cargado en ChatPDF.")

    with st.chat_message("assistant"):
        st.markdown("""
        **Bienvenido a ChatBali**. Puedes preguntar cosas como:
        - ¿Cómo se calcula el subsidio complementario?
        - ¿Qué pasa si el concesionario no cumple?
        - ¿Dónde se describe el indicador de sobredemanda?
        """)

    user_input = st.chat_input("Tu consulta sobre el contrato...")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🔄 Consultando el documento..."):
            headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
            data = {"sourceId": SOURCE_ID, "messages": [{"role": "user", "content": user_input}]}
            response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)

            if response.status_code == 200:
                with st.chat_message("assistant"):
                    st.markdown(response.json()["content"])
            else:
                st.error("❌ No se pudo contactar correctamente a ChatPDF. Revisa la clave o el sourceId.")