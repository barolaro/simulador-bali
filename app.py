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
import re

st.set_page_config(page_title="Simulador Subsidios BALI + Chat", layout="wide")
st.title("📊 Simulador de Subsidios Hospitalarios con Análisis BALI + Chat")

st.markdown("Edita los datos históricos, visualiza la proyección 2025, interpreta automáticamente el comportamiento, descarga el análisis y haz consultas al contrato.")

def proyeccion_y_comentario(nombre_subsidio, valores):
    df = pd.DataFrame({
        "ds": pd.to_datetime([f"{a}-01-01" for a in [2021, 2022, 2023, 2024]]),
        "y": valores
    })
    st.data_editor(df.rename(columns={"ds": "Año", "y": "Monto"}), num_rows="fixed", use_container_width=True)

    x = np.array([2021, 2022, 2023, 2024]).reshape(-1, 1)
    modelo = LinearRegression().fit(x, df["y"])
    pred_lr = modelo.predict([[2025]])[0]

    prophet_df = df.copy()
    modelo_prophet = Prophet()
    modelo_prophet.fit(prophet_df)
    future = modelo_prophet.make_future_dataframe(periods=1, freq='Y')
    forecast = modelo_prophet.predict(future)
    forecast_2025 = forecast[forecast["ds"].dt.year == 2025]
    pred_prophet = forecast_2025["yhat"].values[0] if not forecast_2025.empty else None

    sma = df["y"].rolling(window=2).mean().tolist()

    crecimiento_total = (valores[-1] - valores[0]) / valores[0]
    tasa_anual = ((valores[-1] / valores[0]) ** (1 / (len(valores) - 1))) - 1
    desviacion = np.std(valores)
    tendencia = modelo.coef_[0]

    fig = px.line(df, x="ds", y="y", markers=True, title=f"{nombre_subsidio} - Histórico y Proyección 2025")
    fig.add_scatter(x=[pd.to_datetime("2025-01-01")], y=[pred_lr], mode='markers+text',
                    text=[f"LR: ${pred_lr:,.0f}"], textposition='top right',
                    marker=dict(size=12, color='red'), name="Proy. Lineal")
    if pred_prophet is not None:
        fig.add_scatter(x=[pd.to_datetime("2025-01-01")], y=[pred_prophet], mode='markers+text',
                        text=[f"Prophet: ${pred_prophet:,.0f}"], textposition='bottom left',
                        marker=dict(size=12, color='green'), name="Proy. Prophet")
    st.plotly_chart(fig, use_container_width=True)

    sma_str = f"{sma[-1]:,.0f}" if sma[-1] else "N/A"
    comentario = f"### Análisis Técnico: {nombre_subsidio}\n"
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
    elif desviacion > (0.15 * np.mean(valores)):
        comentario += "\n- Alta variabilidad. Requiere análisis más detallado."
    else:
        comentario += "\n- Estabilidad aceptable. Monitoreo periódico sugerido."

    st.markdown(comentario)

    if st.button(f"📄 Descargar Análisis PDF - {nombre_subsidio}"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        comentario_sin_emojis = re.sub(r'[^
