Simulador BALI Integrado con Análisis y Consulta IA

Este proyecto es una aplicación desarrollada en Streamlit que permite simular el subsidio variable en base a datos históricos, realizar proyecciones para el año 2025 mediante regresión lineal y exponencial, y consultar el documento BALI mediante inteligencia artificial.

🚀 Características

Edición directa de datos históricos del subsidio variable.

Proyección para el año 2025 utilizando modelos lineal y exponencial.

Análisis automático del comportamiento del subsidio según criterios del BALI.

Carga y edición de archivos SFO marzo 2023 y Censo Camas 2022.

Tabla dinámica editable para simulación de escenarios.

Consulta en lenguaje natural al documento BALI usando ChatPDF API.

📂 Estructura del Proyecto

Proyecciones/
├── app.py                  # Código principal de la aplicación Streamlit
├── requirements.txt       # Lista de dependencias
├── SFO marzo 2023.xlsx    # Archivo de entrada para simulación
├── Censo camas 2022.xlsx  # Archivo de referencia para sobredemanda

⚙️ Instalación Local

pip install -r requirements.txt
streamlit run app.py

🔑 Configuración API ChatPDF

Crear un archivo .streamlit/secrets.toml con:

[default]
CHATPDF_API_KEY = "tu_clave_aquí"

📌 Requisitos

Python 3.10+

API Key de ChatPDF

Navegador web moderno

👤 Autor

Bayron RetamalServicio de Salud Metropolitano Occidente (SSMOC)Desarrollado con fines de análisis técnico en salud pública y mejora de la gestión hospitalaria.

📝 Licencia

Este proyecto es de uso interno con fines de simulación y capacitación. Contactar al autor para usos extendidos.

