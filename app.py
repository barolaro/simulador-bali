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

st.markdown("Edita los datos históricos, visualiza la proyección 2025, interpreta automáticamente el comportamiento, descarga el análisis y haz consultas al contrato.")

def remove_non_ascii(text):
    return ''.join(char for char in text if ord(char) < 128)

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
    future = modelo_prophet.make_future_dataframe(periods=1, freq='Y')
    forecast = modelo_prophet.predict(future)
    forecast_2025 = forecast[forecast["ds"].dt.year == 2025]
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
    comentario = f"Análisis Técnico: {nombre_subsidio}\n"
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

    if st.button(f"📄 Descargar Análisis PDF - {nombre_subsidio}"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        comentario_clean = remove_non_ascii(comentario)
        pdf.multi_cell(0, 10, comentario_clean)

        buf = io.BytesIO()
        plt.figure()
        plt.plot(df_editado["ds"].dt.year, df_editado["y"], marker='o', label='Histórico')
        plt.plot(2025, pred_lr, 'ro', label='Proy. Lineal')
        if pred_prophet:
            plt.plot(2025, pred_prophet, 'go', label='Proy. Prophet')
        plt.legend()
        plt.title(nombre_subsidio)
        plt.xlabel("Año")
        plt.ylabel("Monto")
        plt.tight_layout()
        plt.savefig(buf, format="png")
        plt.close()

        buf.seek(0)
        img_path = "grafico_temp.png"
        with open(img_path, "wb") as f:
            f.write(buf.read())
        pdf.image(img_path, x=10, y=pdf.get_y(), w=180)

        pdf_output = f"{nombre_subsidio.replace(' ', '_')}_analisis.pdf"
        pdf.output(pdf_output)
        with open(pdf_output, "rb") as f:
            st.download_button("⬇️ Descargar PDF", f, file_name=pdf_output, mime="application/pdf")

        os.remove(img_path)
        os.remove(pdf_output)

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
