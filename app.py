import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import requests

st.set_page_config(page_title="Simulador Integrado BALI", layout="wide")
st.title("🏥 Simulador Integral: Subsidio, Camas y Consulta BALI")

st.markdown("Una herramienta para proyección de subsidios, simulación de escenarios con camas y consultas inteligentes al documento BALI.")

# ------------------- SUBSIDIO VARIABLE -------------------
with st.expander("📊 Simulador de Subsidio Variable"):
    st.subheader("🔢 Edita los datos históricos")
    subsidio_data = {
        "Año": [2021, 2022, 2023, 2024],
        "Subsidio Variable CLP": [816375829, 2316612803, 1963167525, 2319599141]
    }
    df_subsidio = st.data_editor(pd.DataFrame(subsidio_data), num_rows="fixed", use_container_width=True)

    # Proyección 2025
    años = df_subsidio["Año"].to_numpy()
    valores = df_subsidio["Subsidio Variable CLP"].to_numpy()
    modelo = LinearRegression().fit(años.reshape(-1, 1), valores)
    pred_lineal = modelo.predict(np.array([[2025]]))

    def modelo_exp(x, a, b): return a * np.exp(b * (x - años[0]))
    params, _ = curve_fit(modelo_exp, años, valores, maxfev=10000)
    pred_exponencial = modelo_exp(2025, *params)

    with st.expander("📈 Ver gráfico de proyección detallado"):
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

    crecimiento = (valores[-1] - valores[0]) / valores[0]
    if crecimiento > 1:
        st.success("🔼 Fuerte crecimiento. Verificar Resultados de Servicio (RS) y cumplimiento SIC.")
    elif crecimiento < -0.1:
        st.error("🔽 Caída significativa. Posibles No Conformidades (art. 2.6.2.1 del BALI).")
    else:
        st.info("➡️ Subsidio estable. Revisar correcciones y mantenimientos.")
    st.caption("Referencia: BALI Art. 1.12.2.3 y 2.6.2.1")

# ------------------- SIMULACIÓN CAMAS -------------------
with st.expander("🛏️ Simulación de Escenarios con Camas"):
    st.subheader("📥 Carga archivos SFO y Censo de Camas")
    file_sfo = st.file_uploader("Sube el archivo SFO marzo 2023", type=["xlsx"])
    file_censo = st.file_uploader("Sube el archivo Censo camas 2022", type=["xlsx"])

    if file_sfo and file_censo:
        df_sfo = pd.read_excel(file_sfo)
        df_censo = pd.read_excel(file_censo)

        st.markdown("### 🧾 Censo de Camas 2022 (Editable)")
        df_censo_editado = st.data_editor(df_censo, use_container_width=True)

        st.markdown("### 📉 Resultado simulado")
        df_censo_editado["Camas Proyectadas"] = df_censo_editado.iloc[:, 1] * 1.05  # ejemplo: crecimiento del 5%
        st.dataframe(df_censo_editado)

# ------------------- CHAT INTELIGENTE BALI -------------------
with st.expander("🤖 Consulta Inteligente al BALI"):
    st.subheader("💬 Pregunta al documento BALI")

    API_KEY = st.secrets["CHATPDF_API_KEY"]
    SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"

    with st.chat_message("assistant"):
        st.markdown("""
        **Bienvenido al Asistente de Consulta BALI.**
        Puedes preguntar sobre requisitos del proyecto, subsidios, roles del concesionario, etc.
        """)

    user_input = st.chat_input("Escribe tu duda sobre el contrato...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("🔄 Consultando el documento..."):
            headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
            data = {
                "sourceId": SOURCE_ID,
                "messages": [{ "role": "user", "content": user_input }]
            }
            response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)

            if response.status_code == 200:
                result = response.json()
                with st.chat_message("assistant"):
                    st.markdown(result["content"])
            else:
                st.error("❌ Error en la API. Verifica tu clave o sourceId.")
