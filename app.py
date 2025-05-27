import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from prophet import Prophet
import matplotlib.pyplot as plt
from fpdf import FPDF
import io
import os
import requests

st.set_page_config(page_title="Simulador Subsidios BALI + Chat", layout="wide")
st.title("📊 Simulador de Subsidios Hospitalarios con Análisis BALI + Chat")

def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

def obtener_comentario_chatbali(nombre_subsidio):
    try:
        headers = {"x-api-key": st.secrets["CHATPDF_API_KEY"], "Content-Type": "application/json"}
        prompt = f"Explica en máximo 100 palabras el subsidio '{nombre_subsidio}' según el contrato BALI, indicando su definición, condiciones y aplicación."
        data = {"sourceId": "cha_G85wPwqQ0gYG0SodoZPlh", "messages": [{"role": "user", "content": prompt}]}
        response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)
        if response.status_code == 200:
            return response.json()["content"]
        else:
            return "❌ No se pudo obtener el comentario del contrato BALI."
    except Exception as e:
        return f"❌ Error al conectar con ChatBali: {e}"

def proyeccion_y_comentario(nombre_subsidio, valores_iniciales):
    df = pd.DataFrame({
        "Año": [2021, 2022, 2023, 2024],
        "Monto": valores_iniciales
    })
    df_editado = st.data_editor(df, num_rows="fixed", use_container_width=True)
    df_editado["ds"] = pd.to_datetime(df_editado["Año"].astype(str) + "-01-01")
    df_editado = df_editado.rename(columns={"Monto": "y"})[["ds", "y"]]

    x = np.array([fecha.year for fecha in df_editado["ds"]]).reshape(-1, 1)
    modelo = LinearRegression().fit(x, df_editado["y"])
    pred_lr = modelo.predict([[2025]])[0]

    modelo_prophet = Prophet()
    modelo_prophet.fit(df_editado)

    future = pd.DataFrame({
        "ds": pd.date_range(start=f"{df_editado['ds'].dt.year.min()}-01-01", periods=6, freq="Y")
    })

    forecast = modelo_prophet.predict(future)
    forecast["year"] = forecast["ds"].dt.year
    forecast_2025 = forecast[forecast["year"] == 2025]
    pred_prophet = forecast_2025["yhat"].values[0] if not forecast_2025.empty else None

    sma = df_editado["y"].rolling(window=2).mean().tolist()
    crecimiento_total = (df_editado["y"].iloc[-1] - df_editado["y"].iloc[0]) / df_editado["y"].iloc[0]
    tasa_anual = ((df_editado["y"].iloc[-1] / df_editado["y"].iloc[0]) ** (1 / (len(df_editado) - 1))) - 1
    desviacion = np.std(df_editado["y"])
    tendencia = modelo.coef_[0]

    fig = px.line(df_editado, x="ds", y="y", markers=True, title=f"{nombre_subsidio} - Histórico y Proyección 2025")
    fig.add_scatter(x=[pd.to_datetime("2025-01-01")], y=[pred_lr], mode='markers+text',
                    text=[f"LR: ${pred_lr:,.0f}"], textposition='top right',
                    marker=dict(size=12, color='red'), name="Proy. Lineal")
    if pred_prophet is not None:
        fig.add_scatter(x=[pd.to_datetime("2025-01-01")], y=[pred_prophet], mode='markers+text',
                        text=[f"Prophet: ${pred_prophet:,.0f}"], textposition='bottom left',
                        marker=dict(size=12, color='green'), name="Proy. Prophet")
    st.plotly_chart(fig, use_container_width=True)

    sma_str = f"{sma[-1]:,.0f}" if sma[-1] else "N/A"
    comentario = f"**📈 Análisis Técnico: {nombre_subsidio}**\n"
    comentario += f"- Proyección 2025 (Lineal): ${pred_lr:,.0f}\n"
    comentario += f"{'- Proyección 2025 (Prophet): $' + f'{pred_prophet:,.0f}' if pred_prophet else '- No se generó proyección Prophet para 2025.'}\n"
    comentario += f"- Tasa de crecimiento anual (CAGR): {tasa_anual*100:.2f}%\n"
    comentario += f"- Volatilidad (Desviación estándar): ${desviacion:,.0f}\n"
    comentario += f"- Tendencia: {'Positiva' if tendencia > 0 else 'Negativa'}\n"
    comentario += f"- SMA últimos 2 años: {sma_str}"

    if crecimiento_total > 0.3 and tasa_anual > 0.1:
        comentario += "\n- Crecimiento sostenido. Evaluar relación con metas contractuales."
    elif crecimiento_total < -0.1:
        comentario += "\n- Caída importante. Verificar cumplimiento del BALI."
    elif desviacion > (0.15 * np.mean(df_editado["y"])):
        comentario += "\n- Alta variabilidad. Requiere análisis más detallado."
    else:
        comentario += "\n- Estabilidad aceptable. Monitoreo periódico sugerido."

    st.markdown(comentario)

    with st.spinner("🧠 Generando interpretación del contrato BALI..."):
        comentario_chatbali = obtener_comentario_chatbali(nombre_subsidio)
        st.info(f"📘 Interpretación BALI: {comentario_chatbali}")

tabs = st.tabs([
    "Subsidio Fijo", "Subsidio Variable", "Sobredemanda de Camas",
    "Subsidio Alimentación Adicional", "🤖 ChatBali"
])

with tabs[0]:
    st.subheader("Subsidio Fijo")
    proyeccion_y_comentario("Subsidio Fijo", [1000000000, 1020000000, 1040000000, 1060000000])

with tabs[1]:
    st.subheader("Subsidio Variable")
    proyeccion_y_comentario("Subsidio Variable", [816375829, 2316612803, 1963167525, 2319599141])

with tabs[2]:
    st.subheader("Sobredemanda de Camas")
    proyeccion_y_comentario("Sobredemanda de Camas", [600000000, 630000000, 615000000, 640000000])

with tabs[3]:
    st.subheader("Subsidio Alimentación Adicional")
    proyeccion_y_comentario("Subsidio Alimentación Adicional", [120000000, 130000000, 125000000, 128000000])

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
            headers = {"x-api-key": st.secrets["CHATPDF_API_KEY"], "Content-Type": "application/json"}
            data = {
                "sourceId": "cha_G85wPwqQ0gYG0SodoZPlh",
                "messages": [{"role": "user", "content": user_input}]
            }
            response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)

            if response.status_code == 200:
                with st.chat_message("assistant"):
                    st.markdown(response.json()["content"])
            else:
                st.error("❌ No se pudo contactar correctamente a ChatPDF. Revisa la clave o el sourceId.")
