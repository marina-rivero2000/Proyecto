# Home.py

import streamlit as st
from PIL import Image
import base64
from io import BytesIO


st.set_page_config(page_title="Inicio", layout="centered")

# Cargar imagen local y convertirla a base64
image = Image.open("logo.png")  # Asegúrate que esté en la misma carpeta que este script
buffered = BytesIO()
image.save(buffered, format="PNG")
img_b64 = base64.b64encode(buffered.getvalue()).decode()

st.markdown(
    """
    <style>     
        .sub-header {
            font-size: 18px;
            color: #555;
        }
    </style>
""", unsafe_allow_html=True)

# Cargar imagen y convertirla a base64
def get_base64_image(image_path):
    img = Image.open(image_path)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

img_b64 = get_base64_image("logo.png")

st.markdown(f"""
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <h1 style="color: #004d87 ;">🏠 Bienvenid@</h1>
        <img src="data:image/jpeg;base64,{img_b64}" width="150">
    </div>
""", unsafe_allow_html=True)

st.markdown("""
Usa el menú de la izquierda para navegar a:

<div class="sub-header">
- Recomendador: <br> Selecciona uno o más <strong>artículos vendidos</strong> para obtener sugerencias de productos relacionados que suelen comprarse juntos.

Ideal para:
<ul>
    <li>Sugerir productos complementarios al cliente</li>
    <li>Preparar pedidos completos</li>
    <li>Agilizar la atención en mostrador</li>
</ul>

- Predictor: <br> Selecciona un <strong>artículo</strong> para predecir su demanda futura.

Ideal para:
<ul>
    <li>Planificar compras y reposición</li>
    <li>Optimizar inventario</li>
    <li>Reducir faltantes y sobrestock</li>
</div>
                        
---

💡 Esta herramienta está diseñada para mejorar la atención al cliente y optimizar la venta en tu pequeño comercio.
""", unsafe_allow_html=True)
