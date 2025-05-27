
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit
import requests

st.set_page_config(page_title="Simulador de Subsidios Hospitalarios - BALI", layout="wide")

st.title("📊 Simulador de Subsidios Hospitalarios")
st.caption("Proyecciones, simulación y análisis automático según el documento BALI")

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

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(años, valores, 'o', label="Histórico")
        ax.plot(np.append(años, 2025), modelo.predict(np.append(años, 2025).reshape(-1, 1)), '--', label="Lineal")
        ax.plot(np.append(años, 2025), modelo_exp(np.append(años, 2025), *params), '--', label="Exponencial")
        ax.plot(2025, pred_lineal, 'ro', label=f"2025 (L): ${pred_lineal[0]:,.0f}")
        ax.plot(2025, pred_exponencial, 'bo', label=f"2025 (E): ${pred_exponencial:,.0f}")
        ax.set_ylabel(label_y)
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

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
