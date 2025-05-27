import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
import requests

st.set_page_config(page_title="Simulador BALI + Chat", layout="wide")

st.title("📊 Simulador Integral de Subsidios BALI")

TABS = ["Subsidio Fijo", "Subsidio Variable", "Sobredemanda Camas", "Alimentación Adicional", "ChatBALI"]
tab = st.sidebar.radio("Navegación", TABS)

def mostrar_grafico_y_analisis(nombre, df):
    st.markdown(f"### 💡 Simulación para: {nombre}")
    df = st.data_editor(df, num_rows="dynamic", use_container_width=True)

    if len(df) < 2:
        st.warning("⚠️ Ingrese al menos dos años para proyectar.")
        return

    años = df["Año"].to_numpy()
    valores = df["Monto"].to_numpy()
    modelo = LinearRegression().fit(años.reshape(-1, 1), valores)
    pred = modelo.predict(np.array([[2025]]))[0]

    fig = px.line(df, x="Año", y="Monto", markers=True, title=f"{nombre} - Histórico y Proyección")
    fig.add_scatter(x=[2025], y=[pred], mode='markers+text',
                    marker=dict(color="red", size=10),
                    text=[f"Proy: ${pred:,.0f}"], textposition="top center")

    st.plotly_chart(fig, use_container_width=True)

    # Comentario basado en crecimiento
    crecimiento = (valores[-1] - valores[0]) / valores[0]
    if crecimiento > 1:
        st.success("📈 Crecimiento importante. Revisar Art. 1.12.2.3 del BALI sobre cumplimiento progresivo.")
    elif crecimiento < -0.1:
        st.error("📉 Disminución significativa. Puede haber impacto en RS o penalidades. Ver art. 2.6.2.1.")
    else:
        st.info("➡️ Comportamiento estable. Seguir monitoreando indicadores según el BALI.")

if tab == "Subsidio Fijo":
    df_fijo = pd.DataFrame({
        "Año": [2021, 2022, 2023, 2024],
        "Monto": [1000000000, 1020000000, 1040000000, 1060000000]
    })
    mostrar_grafico_y_analisis("Subsidio Fijo", df_fijo)

elif tab == "Subsidio Variable":
    df_variable = pd.DataFrame({
        "Año": [2021, 2022, 2023, 2024],
        "Monto": [816375829, 2316612803, 1963167525, 2319599141]
    })
    mostrar_grafico_y_analisis("Subsidio Variable", df_variable)

elif tab == "Sobredemanda Camas":
    df_sobredemanda = pd.DataFrame({
        "Año": [2021, 2022, 2023, 2024],
        "Monto": [12000000, 13500000, 11000000, 15500000]
    })
    mostrar_grafico_y_analisis("Sobredemanda Camas", df_sobredemanda)

elif tab == "Alimentación Adicional":
    df_alimentacion = pd.DataFrame({
        "Año": [2021, 2022, 2023, 2024],
        "Monto": [20000000, 21000000, 19000000, 25000000]
    })
    mostrar_grafico_y_analisis("Alimentación Adicional", df_alimentacion)

elif tab == "ChatBALI":
    st.header("💬 Consulta el Contrato BALI con IA")
    st.markdown("Puedes preguntar directamente sobre artículos del contrato BALI cargado en ChatPDF.")

    API_KEY = st.secrets["CHATPDF_API_KEY"]
    SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"  # ID válido de ChatPDF para contrato.pdf

    user_input = st.chat_input("Escribe tu duda sobre el contrato...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("⏳ Consultando contrato BALI..."):
            headers = {
                "x-api-key": API_KEY,
                "Content-Type": "application/json"
            }
            data = {
                "sourceId": SOURCE_ID,
                "messages": [
                    { "role": "user", "content": user_input }
                ]
            }
            response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)

            if response.status_code == 200:
                result = response.json()
                with st.chat_message("assistant"):
                    st.markdown(result["content"])
            else:
                st.error("⚠️ Error en la API. Revisa tu clave o el ID del documento.")
