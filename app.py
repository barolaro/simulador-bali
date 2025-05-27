
import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Simulador Subsidios BALI", layout="wide")
st.title("📊 Simulador BALI: Proyección de Subsidios y Consulta Inteligente")

# --- Datos base ---
df = pd.DataFrame({
    "Año": [2020, 2021, 2022, 2023, 2024],
    "Subsidio Fijo": [3182836153, 6789450876, 7067472653, 8106304345, 12075185403],
    "Subsidio Variable": [0, 816375829, 2316612803, 1963167525, 2319599141],
    "Sobredemanda Camas": [0, 673988.81, 90545974, 33401975, 3076438.79],
    "Alimentación Adicional": [254466335, 118404126, 263997309, 584616195, 474433993]
})

subsidios = [col for col in df.columns if col != "Año"]
tabs = st.tabs([f"📌 {s}" for s in subsidios] + ["💬 ChatBALI"])

# --- Función de proyección ---
def proyectar_2025(col):
    modelo = LinearRegression().fit(df[["Año"]], df[col])
    return modelo.predict([[2025]])[0]

# --- Renderizar cada subsidio ---
for i, subsidio in enumerate(subsidios):
    with tabs[i]:
        st.subheader(f"📊 {subsidio}")
        st.dataframe(df[["Año", subsidio]], use_container_width=True)

        fig = px.line(df, x="Año", y=subsidio, markers=True, title=f"Proyección {subsidio}")
        y_pred = proyectar_2025(subsidio)
        fig.add_scatter(x=[2025], y=[y_pred], mode="markers+text",
                        text=[f"${y_pred:,.0f}"], name="2025")
        st.plotly_chart(fig, use_container_width=True)

        # Análisis automático
        delta = df[subsidio].iloc[-1] - df[subsidio].iloc[-2]
        if delta > 0:
            st.success(f"📈 Aumento respecto al año anterior: ${delta:,.0f}")
        elif delta < 0:
            st.warning(f"📉 Disminución respecto al año anterior: ${delta:,.0f}")
        else:
            st.info("⏸️ Sin variación respecto al año anterior.")

# --- Chat interactivo simulado ---
with tabs[-1]:
    st.subheader("💬 Chat BALI Interactivo")

    if "mensajes" not in st.session_state:
        st.session_state["mensajes"] = []

    for m in st.session_state["mensajes"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    pregunta = st.chat_input("Escribe tu consulta sobre el contrato o subsidios BALI...")

    if pregunta:
        st.session_state["mensajes"].append({"role": "user", "content": pregunta})
        with st.chat_message("user"):
            st.markdown(pregunta)

        respuesta = "📘 Revisa el Art. 2.6.2.1 y 1.12.2.3 del BALI para más detalles sobre esta sección."
        st.session_state["mensajes"].append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)