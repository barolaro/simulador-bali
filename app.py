
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

st.title("📈 Proyección y Cálculo de Subsidios (SFO)")
st.markdown("Herramienta dinámica para estimar valores ajustados y proyectar financieramente con distintos modelos.")

# --- FUNCIONES ---
@st.cache_data
def obtener_valor_uf():
    try:
        response = requests.get("https://mindicador.cl/api")
        data = response.json()
        return float(data['uf']['valor'])
    except:
        return None

@st.cache_data
def obtener_sueldo_minimo():
    try:
        response = requests.get("https://www.dt.gob.cl/portal/1628/w3-article-60141.html")
        soup = BeautifulSoup(response.text, 'html.parser')
        for p in soup.find_all('p'):
            if 'sueldo mínimo' in p.text.lower():
                valor = p.text.split()[-1].replace('$', '').replace('.', '').strip()
                return int(valor)
        return 500000
    except:
        return 500000

# --- PARTE 1: CÁLCULO SFO ---
st.subheader("🔢 Cálculo de Subsidio Fijo (SFO)")
uf = obtener_valor_uf()
sueldo = obtener_sueldo_minimo()

if uf and sueldo:
    # Parámetros
    sfo_base = 242122
    w_min_0, w_min_j1 = 9.15, 11.64
    ipa_0, ipa_j1 = 81.97, 149.56
    f1, f2 = 0.05, 0.23

    sfo_aj = sfo_base * ((1 - f1 - f2) + f1 * (ipa_j1 / ipa_0) + f2 * (w_min_j1 / w_min_0))
    sfo_clp = sfo_aj * uf
    iva = sfo_clp * 0.19
    total = sfo_clp + iva

    st.metric("💸 SFO Ajustado (UF)", f"{sfo_aj:,.2f}")
    st.metric("💰 SFO CLP sin IVA", f"${sfo_clp:,.0f}")
    st.metric("🧾 IVA (19%)", f"${iva:,.0f}")
    st.metric("📊 Total con IVA", f"${total:,.0f}")
else:
    st.error("No se pudieron obtener los valores actualizados de UF o sueldo mínimo.")

# --- PARTE 2: PROYECCIÓN ---
st.subheader("📉 Proyección Financiera 2020–2025")

años = np.array([2020, 2021, 2022, 2023, 2024])
valores = np.array([3182836153, 6789450876, 7067472653, 8106304345, 12075185403])

# Modelos
modelo_lineal = LinearRegression().fit(años.reshape(-1, 1), valores)
pred_lineal = modelo_lineal.predict(np.array([[2025]]))

def modelo_exponencial(x, a, b):
    return a * np.exp(b * (x - 2020))

params, _ = curve_fit(modelo_exponencial, años, valores)
pred_exponencial = modelo_exponencial(2025, *params)

# Gráfico
fig, ax = plt.subplots()
ax.plot(años, valores, 'o', label="Datos históricos")
ax.plot(años, modelo_lineal.predict(años.reshape(-1, 1)), '--', label="Lineal")
ax.plot(np.append(años, 2025), modelo_exponencial(np.append(años, 2025), *params), '--', label="Exponencial")
ax.plot(2025, pred_lineal, 'ro', label=f"Lineal 2025: ${pred_lineal[0]:,.0f}")
ax.plot(2025, pred_exponencial, 'bo', label=f"Exponencial 2025: ${pred_exponencial:,.0f}")
ax.set_title("Proyección Financiera")
ax.set_xlabel("Año")
ax.set_ylabel("Valor en CLP")
ax.grid(True)
ax.legend()
st.pyplot(fig)



# --- Simulación Interactiva ---
st.markdown("### 📊 Simulación de Proyección")

col1, col2 = st.columns(2)
max_año = col1.slider("Año máximo de simulación", min_value=2025, max_value=2035, value=2030)
crecimiento = col2.slider("Tasa de crecimiento anual (%)", min_value=0, max_value=50, value=10)

st.info(f"Proyectando hasta {max_año} con un crecimiento del {crecimiento}% anual")

df_simulado = df.copy()
años = list(range(df["Año"].min(), max_año + 1))

df_resultado = {"Año": años}
for tipo in ["Fijo", "Variable", "Alimentación"]:
    base = df[df["Año"] == df["Año"].max()][tipo].values[0]
    proy = [base * ((1 + crecimiento/100) ** (año - df["Año"].max())) for año in años]
    df_resultado[tipo] = proy

df_proy = pd.DataFrame(df_resultado)

fig_sim = go.Figure()
for tipo in ["Fijo", "Variable", "Alimentación"]:
    fig_sim.add_trace(go.Scatter(x=df_proy["Año"], y=df_proy[tipo], mode="lines+markers", name=tipo))

fig_sim.update_layout(title="Simulación de Proyección de Subsidios", xaxis_title="Año", yaxis_title="Monto Proyectado")
st.plotly_chart(fig_sim, use_container_width=True)



# --- ChatBali al final ---
st.markdown("---")
st.markdown("### 🤖 ChatBali: Consulta tu Documento BALI")

st.markdown("Haz preguntas como:")
st.markdown("- ¿Qué obligaciones tiene el concesionario?")
st.markdown("- ¿Qué dice la cláusula 1.12.2.2 sobre nutrición clínica?")

API_KEY = st.secrets["CHATPDF_API_KEY"]
SOURCE_ID = "cha_G85wPwqQ0gYG0SodoZPlh"

user_input = st.chat_input("Escribe tu pregunta sobre las BALI...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    headers = {"x-api-key": API_KEY, "Content-Type": "application/json"}
    data = {"sourceId": SOURCE_ID, "messages": [{"role": "user", "content": user_input}]}
    response = requests.post("https://api.chatpdf.com/v1/chats/message", json=data, headers=headers)

    if response.status_code == 200:
        result = response.json()
        with st.chat_message("assistant"):
            st.markdown(result["content"])
    else:
        st.error("❌ Error al conectar con ChatPDF")
