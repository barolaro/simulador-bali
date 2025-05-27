import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import requests

st.set_page_config(page_title="Simulador + Análisis BALI", layout="wide")
st.title("📊 Simulador de Subsidio Variable con Análisis BALI + Chat")

st.markdown("Edita los datos históricos, visualiza la proyección 2025 y consulta directamente al documento BALI con inteligencia artificial.")

# --- Sección 1: Ingreso de Datos ---
with st.expander("🔢 Edita los datos históricos"):
    default_data = {
        "Año": [2021, 2022, 2023, 2024],
        "Subsidio Variable CLP": [816375829, 2316612803, 1963167525, 2319599141]
    }
    df = st.data_editor(pd.DataFrame(default_data), num_rows="fixed", use_container_width=True)

# --- Sección 2: Gráfico Compacto ---
with st.expander("📊 Ver gráfico de proyección detallado"):
    años = df["Año"].to_numpy()
    valores = df["Subsidio Variable CLP"].to_numpy()
    años_reshape = años.reshape(-1, 1)
    modelo = LinearRegression().fit(años_reshape, valores)
    pred_lineal = modelo.predict(np.array([[2025]]))

    def modelo_exp(x, a, b): return a * np.exp(b * (x - años[0]))
    params, _ = curve_fit(modelo_exp, años, valores, maxfev=10000)
    pred_exponencial = modelo_exp(2025, *params)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(años, valores, 'o', label="Histórico")
    ax.plot(np.append(años, 2025), modelo.predict(np.append(años, 2025).reshape(-1, 1)), '--', label="Lineal")
    ax.plot(np.append(años, 2025), modelo_exp(np.append(años, 2025), *params), '--', label="Exponencial")
    ax.plot(2025, pred_lineal, 'ro', label=f"2025 (L): ${pred_lineal[0]:,.0f}")
    ax.plot(2025, pred_exponencial, 'bo', label=f"2025 (E): ${pred_exponencial:,.0f}")
    ax.set_ylabel("CLP")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# --- Sección 3: Análisis Dinámico ---
st.subheader("🧠 Análisis automático según BALI")
crecimiento = (valores[-1] - valores[0]) / valores[0]

if crecimiento > 1:
    st.success("🔼 Fuerte crecimiento en subsidio variable. Probable mejora en Resultados de Servicio (RS). Verificar cumplimiento real en sistema SIC.")
elif crecimiento < -0.1:
    st.error("🔽 Caída significativa. Indica No Conformidades no corregidas según art. 2.6.2.1 del BALI.")
else:
    st.info("➡️ Subsidio relativamente estable. Mantener control sobre indicadores y correcciones del SIC.")

st.caption("Referencia: BALI Art. 1.12.2.3 y 2.6.2.1")

# --- Sección 4: Chat para Consultas BALI ---
st.divider()
st.subheader("💬 Pregunta al documento BALI")

st.markdown("Puedes consultar directamente el contenido del documento de bases de licitación para resolver dudas.")

API_KEY = st.secrets["CHATPDF_API_KEY"]
SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"

# Mensaje de bienvenida
with st.chat_message("assistant"):
    st.markdown("""
**Bienvenido al Asistente de Consulta BALI.**
Puedes preguntar sobre requisitos del proyecto, subsidios, roles del concesionario, indicadores de servicio, etc.
Ejemplos:
- ¿Qué significa el indicador RS?
- ¿Cómo se calcula el subsidio variable?
- ¿Qué artículos regulan la sobredemanda?
    """)

# Entrada usuario
user_input = st.chat_input("Escribe tu duda sobre el contrato...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.spinner("🔄 Consultando el documento..."):
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
            st.error("❌ Error en la API. Verifica tu clave o sourceId.")
