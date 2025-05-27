
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import requests

st.set_page_config(page_title="Simulador BALI", layout="wide", page_icon="📊")

st.markdown("<h1 style='text-align: center; color: #1E90FF;'>📊 Simulador BALI: Subsidios y Análisis</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Visualización y consulta inteligente del documento BALI</p>", unsafe_allow_html=True)

tabs = st.tabs(["Subsidio Fijo", "Subsidio Variable", "Sobredemanda de Camas", "Alimentación", "📄 ChatBALI"])

def editor_y_grafico(nombre, años_base, valores_base, base_year, label_y, interpretador):
    st.subheader(f"✏️ Editar datos históricos - {nombre}")
    df = st.data_editor(pd.DataFrame({"Año": años_base, f"{nombre} CLP": valores_base}), use_container_width=True)
    años = df["Año"].to_numpy()
    valores = df[df.columns[1]].to_numpy()

    st.subheader("📉 Proyección al 2025")
    with st.expander("📊 Ver gráfico"):
        años_reshape = años.reshape(-1, 1)
        modelo = LinearRegression().fit(años_reshape, valores)
        pred_lineal = modelo.predict(np.array([[2025]]))
        def modelo_exp(x, a, b): return a * np.exp(b * (x - base_year))
        params, _ = curve_fit(modelo_exp, años, valores, maxfev=10000)
        pred_exponencial = modelo_exp(2025, *params)

        
fig = go.Figure()
fig.add_trace(go.Scatter(x=años, y=valores, mode="lines+markers", name="Histórico"))
fig.add_trace(go.Scatter(x=[2025], y=[modelo.predict([[2025]])[0]], mode="markers", name="2025 (Lineal)", marker=dict(color="red", size=10)))
fig.add_trace(go.Scatter(x=[2025], y=[modelo_exp(2025, *params)], mode="markers", name="2025 (Exponencial)", marker=dict(color="blue", size=10)))
fig.update_layout(height=300, margin=dict(t=10, b=10), xaxis_title="Año", yaxis_title=label_y)
st.plotly_chart(fig, use_container_width=True)


st.subheader("🧠 Análisis automático")
    interpretador(valores)

# Interpreta tendencias
def interpretar_sfo(valores):
    growth = (valores[-1] - valores[0]) / valores[0]
    if growth > 0.5:
        st.warning("⚠️ Crecimiento acelerado del subsidio fijo. Requiere ajustes financieros.")
    else:
        st.success("✅ Subsidio fijo con crecimiento controlado.")
    st.caption("Referencia: BALI Art. 1.12.2.2")

def interpretar_variable(valores):
    change = (valores[-1] - valores[0]) / valores[0]
    if change > 1:
        st.success("🔼 Mejoras en Resultados de Servicio. Verificar cumplimiento real en SIC.")
    elif change < -0.1:
        st.error("🔽 Caída importante. Indica incumplimientos. Revisar alertas SIC.")
    else:
        st.info("➡️ Subsidio variable estable.")
    st.caption("Referencia: BALI Art. 1.12.2.3 y 2.6.2.1")

def interpretar_sobredemanda(valores):
    if max(valores) > 1_000_000:
        st.warning("⚠️ Altos pagos por sobredemanda. Revisar capacidad instalada.")
    else:
        st.success("✅ Sobredemanda bajo control.")
    st.caption("Referencia: BALI Art. 1.12.2.6")

def interpretar_alimentacion(valores):
    if valores[-1] > 1.2 * np.mean(valores[:-1]):
        st.warning("⚠️ Aumento atípico. Posible error de planificación o crisis.")
    else:
        st.info("✅ Alimentación adicional dentro de rangos normales.")
    st.caption("Referencia: BALI Art. 1.12.2.4")

# Tabs individuales
with tabs[0]:
    editor_y_grafico("Subsidio Fijo", [2020, 2021, 2022, 2023, 2024],
                     [3182836153, 6789450876, 7067472653, 8106304345, 12075185403],
                     2020, "CLP", interpretar_sfo)

with tabs[1]:
    editor_y_grafico("Subsidio Variable", [2021, 2022, 2023, 2024],
                     [816375829, 2316612803, 1963167525, 2319599141],
                     2021, "CLP", interpretar_variable)

with tabs[2]:
    editor_y_grafico("Sobredemanda Camas", [2021, 2022, 2023, 2024],
                     [673988.81, 90545974, 33401975, 3076438.79], 2021, "CLP ajustado", interpretar_sobredemanda)

with tabs[3]:
    editor_y_grafico("Alimentación Adicional", [2020, 2021, 2022, 2023, 2024],
                     [254466335, 118404126, 263997309, 584616195, 474433993],
                     2020, "CLP", interpretar_alimentacion)

# ChatBALI
with tabs[4]:
    st.subheader("💬 Asistente de Consulta BALI")
    st.markdown("Pregunta directamente sobre el contrato de licitación.")

    API_KEY = st.secrets["CHATPDF_API_KEY"]
    SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"

    with st.chat_message("assistant"):
        st.markdown("""
        **Bienvenido a ChatBALI.**
        Puedes consultar el contenido de las bases de licitación.
        Ejemplos:
        - ¿Cómo se calcula el subsidio variable?
        - ¿Qué pasa si se supera el 110% de camas?
        - ¿Qué artículos regulan la alimentación adicional?
        """)

    user_input = st.chat_input("Escribe tu consulta...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.spinner("⏳ Consultando documento BALI..."):
            headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
            data = {
                "sourceId": SOURCE_ID,
                "messages": [{"role": "user", "content": user_input}]
            }

            response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)

            if response.status_code == 200:
                result = response.json()
                with st.chat_message("assistant"):
                    st.markdown(result["content"])
            else:
                st.error("❌ Error al conectar con ChatPDF.")
