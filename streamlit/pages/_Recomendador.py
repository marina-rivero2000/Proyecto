import streamlit as st
from models.modelo_recomendador import cargar_datos, entrenar_modelo, generar_reglas, recomendar_por_productos
import pandas as pd
from PIL import Image
import base64
from io import BytesIO

# Configurar la página
st.set_page_config(page_title="🔧 Recomendador para Ferretería", layout="centered")

# Logo y título
st.markdown("""
<style>     
    .sub-header {
        font-size: 18px;
        color: #555;
    }
            
    .recomendacion {
        background-color: #e8f4f8;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 5px;
        border-left: 5px solid #2E86C1;
    }
            
    /* Aplicar a todos los botones de Streamlit */
    .stButton > button {
        border: 1px solid;
        color: black;
        background-color: white;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        color: #2E86C1 !important;           /* texto azul */
        border: 1px solid #2E86C1 !important; /* borde azul */
        background-color: #f0faff !important; /* fondo opcional celeste muy claro */
    }

    /* Estado focus (al hacer clic y mantenerlo) */
    .stButton > button:focus {
        box-shadow: black !important;
        color: black !important;
        border: 1px solid #2E86C1 !important;
        background-color: #f0faff !important;
    }
            
    /* Burbujas de artículos seleccionados */
    [data-baseweb="tag"] {
        background-color: white !important;
        color: black !important;
        font-weight: 600;
        border-radius: 5px;
        border: 1px solid white !important; /* borde azul */
    }

    /* Ícono de eliminar en las burbujas */
    [data-baseweb="tag"] svg {
        color: black !important;
    }
            
    /* Aplicar solo dentro del contenedor .custom-multiselect */
    [data-baseweb="select"] {
        border: 1px solid !important;
        color: grey !important;
        border-radius: 6px !important;
        transition: all 0.3s ease;
    }
    /* Evitar cambio de borde al hacer clic o enfocar */
    [data-baseweb="select"].is-focused > div,
    [data-baseweb="select"]:focus-within > div,
    [data-baseweb="select"][aria-invalid="true"] > div {
        border: 1px solid #ccc !important;
        box-shadow: none !important;
    }
    /* Forzar siempre borde gris al multiselect */
    [data-baseweb="select"] > div {
        border: 1px solid #ccc !important;       /* Gris */
        box-shadow: none !important;
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
        <h1 style="color: #004d87;">Recomendador de Artículos</h1>
        <img src="data:image/jpeg;base64,{img_b64}" width="145">
    </div>
""", unsafe_allow_html=True)

# Inicializar estado
if "articulos_seleccionados" not in st.session_state:
    st.session_state.articulos_seleccionados = []
if "recomendaciones" not in st.session_state:
    st.session_state.recomendaciones = []

# Funciones
def limpiar_seleccion():
    st.session_state.articulos_seleccionados = []
    st.session_state.recomendaciones = []

def recomendar():
    cesta = st.session_state.articulos_seleccionados
    if cesta:
        st.session_state.recomendaciones = recomendar_por_productos(
            cesta, ventas, reglas_estrictas, reglas_no_estrictas, populares, top_n=3
        )
    else:
        st.warning("⚠️ Por favor, selecciona al menos un artículo.")

# Cargar datos
ventas = cargar_datos()
frequency_ap, df_train = entrenar_modelo(ventas, soporte_minimo=0.02)
reglas_estrictas = generar_reglas(frequency_ap, lift_min=2, conf_min=0.5)
reglas_no_estrictas = generar_reglas(frequency_ap, lift_min=1, conf_min=0.3)
populares = df_train.sum().sort_values(ascending=False).index.tolist()
articulos = sorted(ventas['Descripcion_Art'].dropna().astype(str).unique())

st.markdown("#### 🛠️ Selección de artículos")

# Layout
col_sel, col_btns = st.columns([3.3, 2])

with col_sel:
    st.markdown('<div class="custom-multiselect">', unsafe_allow_html=True)
    cesta = st.multiselect(
        "Selecciona los artículos que ha comprado el cliente:",
        articulos,
        default=st.session_state.articulos_seleccionados,
        key="articulos_seleccionados",
        placeholder="Elige uno o más artículos...",
    )
    st.markdown('</div>', unsafe_allow_html=True)

with col_btns:
    # Centrar verticalmente los botones con CSS
    st.markdown(
        """
        <style>
        .btn-container {
            display: flex;
        flex-direction: column;
        justify-content: center;
        height: 100%;
        gap: 10px;
        padding-top: 30px; /* Ajuste visual para alinear mejor */
        }
        </style>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="btn-container">', unsafe_allow_html=True)
    col_a, col_b = st.columns([1.3, 1])
    with col_a:
        st.button("🔍 Recomendar", on_click=recomendar, use_container_width=True)
    with col_b:
        st.button("🗑️ Limpiar", on_click=limpiar_seleccion, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
# Resultados
if st.session_state.recomendaciones:
    st.subheader("📦 Productos recomendados")

    for i, fam in enumerate(st.session_state.recomendaciones, 1):
        st.markdown(f'<div class="recomendacion"> <strong>{i}. {fam}</strong></div>', unsafe_allow_html=True)
