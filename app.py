
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import base64

st.set_page_config(page_title="Simulador BALI", layout="wide", page_icon="📊")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>📊 Simulador BALI: Subsidios y Análisis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Visualización y consulta inteligente del documento BALI</p>", unsafe_allow_html=True)

# --- DATOS ---
años = np.array([2020, 2021, 2022, 2023, 2024]).reshape(-1, 1)
subsidio_fijo = np.array([3182836153, 6789450876, 7067472653, 8106304345, 12075185403])
subsidio_variable = np.array([0, 816375829, 2316612803, 1963167525, 2319599141])
sobredemanda_camas = np.array([0, 673988.81, 90545974, 33401975, 3076438.79])
alimentacion_adicional = np.array([254466335, 118404126, 263997309, 584616195, 474433993])

# --- FUNCIÓN EXPONENCIAL ---
def modelo_exp(x, a, b, c):
    return a * np.exp(b * x) + c

# --- FUNCIÓN DE ANÁLISIS AUTOMÁTICO ---
def interpretador(valores):
    crecimiento = valores[-1] - valores[-2]
    if crecimiento > 0:
        st.success("🔎 El último año muestra un aumento de subsidios respecto al año anterior.")
    elif crecimiento < 0:
        st.warning("📉 Se observa una disminución en los subsidios respecto al año anterior.")
    else:
        st.info("⏸️ No hubo variación entre el último año y el anterior.")

# --- FUNCIÓN DE EXPORTACIÓN EXCEL ---
def generar_excel():
    df = pd.DataFrame({
        "Año": años.flatten(),
        "Subsidio Fijo": subsidio_fijo,
        "Subsidio Variable": subsidio_variable,
        "Sobredemanda Camas": sobredemanda_camas,
        "Alimentación Adicional": alimentacion_adicional
    })
    towrite = pd.ExcelWriter("simulador_export.xlsx", engine="xlsxwriter")
    df.to_excel(towrite, index=False, sheet_name="Subsidios")
    towrite.close()
    with open("simulador_export.xlsx", "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
        href = f'<a href="data:application/octet-stream;base64,{b64}" download="simulador_subsidios.xlsx">📥 Descargar Excel</a>'
        return href

# --- GRÁFICO PLOTLY ---
def graficar(valores, nombre):
    modelo = LinearRegression().fit(años, valores)
    prediccion = modelo.predict([[2025]])

    try:
        params, _ = curve_fit(modelo_exp, años.flatten(), valores)
        prediccion_exp = modelo_exp(2025, *params)
    except:
        prediccion_exp = None

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=años.flatten(), y=valores, mode="lines+markers", name="Histórico"))
    fig.add_trace(go.Scatter(x=[2025], y=[prediccion[0]], mode="markers", name="2025 (Lineal)", marker=dict(color="red", size=10)))
    if prediccion_exp:
        fig.add_trace(go.Scatter(x=[2025], y=[prediccion_exp], mode="markers", name="2025 (Exponencial)", marker=dict(color="blue", size=10)))
    fig.update_layout(title=nombre, height=350, xaxis_title="Año", yaxis_title="Monto", margin=dict(t=30, b=30))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("🧠 Análisis automático")
    interpretador(valores)

# --- LAYOUT ---
col1, col2 = st.columns(2)

with col1:
    graficar(subsidio_fijo, "Subsidio Fijo")

with col2:
    graficar(subsidio_variable, "Subsidio Variable")

col3, col4 = st.columns(2)

with col3:
    graficar(sobredemanda_camas, "Sobredemanda Camas")

with col4:
    graficar(alimentacion_adicional, "Alimentación Adicional")

# --- BOTÓN DESCARGA ---
st.markdown("### 📤 Exportar subsidios")
st.markdown(generar_excel(), unsafe_allow_html=True)