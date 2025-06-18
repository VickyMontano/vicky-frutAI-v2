import streamlit as st
import torch
import numpy as np
import cv2
import os
from PIL import Image, Image as PILImage
from utils import cargar_modelo, segmentar_frutas, clasificar_imagen, descargar_si_falta

st.set_page_config(page_title="FrutAI 🍍", page_icon="🍓")

# 🧠 Forzar CPU en Streamlit
device = torch.device("cpu")

# 📦 Descargar modelo si falta
descargar_si_falta("best_efficientnet_b3.pth", "https://huggingface.co/VickyMontano03/frutai-models/resolve/main/best_efficientnet_b3.pth")

# 🔍 Cargar modelo
model, transform, class_names = cargar_modelo("best_efficientnet_b3.pth", device)

# 🌈 Estilos visuales
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
    color: #eee;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-container {
    background: rgba(255, 255, 255, 0.12);
    border-radius: 16px;
    padding: 25px 40px 40px 40px;
    box-shadow: 0 8px 32px 0 rgba(45, 60, 80, 0.37);
    margin-bottom: 40px;
}
.description {
    font-size: 1.25rem;
    color: #d1c4e9;
    text-align: center;
    margin-bottom: 30px;
}
div.stButton > button {
    background: #7e57c2;
    color: white;
    font-weight: bold;
    border-radius: 10px;
    padding: 10px 25px;
    border: none;
    box-shadow: 0 4px 15px #b39ddb;
}
div.stButton > button:hover {
    background: #512da8;
    box-shadow: 0 6px 20px #311b92;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-container">', unsafe_allow_html=True)
st.image("portada.jpg", use_container_width=True)
st.markdown('<p class="description">Cargá una imagen con frutas y detectamos y clasificamos automáticamente.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Subí una imagen de frutas", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image_np, caption="📷 Imagen original", use_container_width=True)

    with st.spinner("🔍 Segmentando frutas..."):
        boxes = segmentar_frutas(image_np, device)

    with st.spinner("🧠 Clasificando frutas..."):
        resultados = clasificar_imagen(image_np, boxes, model, transform, class_names, device)

    if resultados:
        image_final = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = image_final.shape[:2]
        scale_factor = 1.0
        max_width = 800

        if orig_w > max_width:
            scale_factor = max_width / orig_w
            new_w, new_h = max_width, int(orig_h * scale_factor)
        else:
            new_w, new_h = orig_w, orig_h

        image_final_resized = cv2.resize(image_final, (new_w, new_h))

        for d in resultados:
            x, y, w_box, h_box = d["bbox"]
            label, conf = d["label"], d["conf"]
            x_s, y_s = int(x * scale_factor), int(y * scale_factor)
            w_s, h_s = int(w_box * scale_factor), int(h_box * scale_factor)

            cv2.rectangle(image_final_resized, (x_s, y_s), (x_s + w_s, y_s + h_s), (0, 255, 0), 2)
            text = f"{label} ({conf:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_x = x_s + (w_s - text_w) // 2
            text_y = y_s + (h_s + text_h) // 2
            cv2.putText(image_final_resized, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        st.image(cv2.cvtColor(image_final_resized, cv2.COLOR_BGR2RGB), caption="🍓 Clasificación final")

        # 🍍 Recetas
        recetariouno = {
            "Anana": ["Helado de ananá natural", "Mousse de ananá", "Tarta tropical de frutas"],
            "Banana": ["Banana split", "Helado de banana casero", "Tarta fría de banana"],
            "Coco": ["Helado de coco", "Trufas de coco y chocolate blanco", "Cheesecake de coco"],
            "Frutilla": ["Helado de frutilla", "Tarta helada de frutilla", "Parfait de frutas"],
            "Higo": ["Helado de higos con miel", "Higos rellenos con queso crema", "Postre crocante de higos"],
            "Manzana": ["Manzanas caramelizadas", "Crumble de manzana", "Helado de manzana verde"],
            "Mora": ["Sorbete de mora", "Copa de moras y crema", "Helado de frutos rojos"],
            "Naranja": ["Helado de naranja", "Gelatina cítrica con crema", "Mousse de naranja"],
            "Palta": ["Ensalada de palta con pera", "Palta dulce con yogurt y miel", "Tartitas saladas con palta y frutas"],
            "Pera": ["Helado de pera", "Tarta dulce de peras con nuez", "Peras al horno con crema chantilly"]
        }

        archivo_por_receta = {
            "Helado de ananá natural": "HeladoAnana.jpg",
            "Mousse de ananá": "mousseAnana.jpg",
            "Tarta tropical de frutas": "tortaTropical.jpg",
            "Helado de banana casero": "heladoBananaCasero.jpg",
            "Tarta fría de banana": "tartaFriaBanana.jpg",
            "Helado de coco": "HeladoCoco.jpg",
            "Trufas de coco y chocolate blanco": "TrufasCoco.jpg",
            "Cheesecake de coco": "cheesecakeCoco.jpg",
            "Helado de frutilla": "heladoFrutilla.jpg",
            "Tarta helada de frutilla": "TartaFrutilla.jpg",
            "Parfait de frutas": "Parfait.jpg",
            "Helado de higos con miel": "HeladoHigo.jpg",
            "Higos rellenos con queso crema": "HigoRelleno.jpg",
            "Postre crocante de higos": "CrocanteHigo.jpg",
            "Manzanas caramelizadas": "manzanaCaramelizadas.jpg",
            "Crumble de manzana": "crumble.jpg",
            "Helado de manzana verde": "heladoManzanaVerde.jpg",
            "Sorbete de mora": "sorbeteMora.jpg",
            "Copa de moras y crema": "MoraYCrema.jpg",
            "Helado de frutos rojos": "HeladoFrutosRojos.jpg",
            "Helado de naranja": "heladoNaranja.jpg",
            "Gelatina cítrica con crema": "GelatinaCrema.jpg",
            "Mousse de naranja": "MousseNaranja.jpg",
            "Ensalada de palta con pera": "ensaladaPaltaPera.jpg",
            "Palta dulce con yogurt y miel": "paltaDulce.jpg",
            "Tartitas saladas con palta y frutas": "TartaPalta.jpg",
            "Helado de pera": "HeladoPera.jpg",
            "Tarta dulce de peras con nuez": "TartaPera.jpg",
            "Peras al horno con crema chantilly": "PerasAlHorno.jpg"
        }

        fruta_top = resultados[0]["label"]
        recetas = recetariouno.get(fruta_top, [])
        if recetas:
            st.subheader(f"🍽️ Recetas con {fruta_top}")
            cols = st.columns(len(recetas))
            for i, receta in enumerate(recetas):
                archivo = archivo_por_receta.get(receta)
                if archivo:
                    path = os.path.join("../data", "recetariouno", archivo)
                    try:
                        img = PILImage.open(path)
                        with cols[i]:
                            st.image(img, caption=receta, use_container_width=True)
                    except:
                        st.warning(f"❌ No se pudo cargar: {receta}")
        else:
            st.warning(f"❌ No hay recetas para {fruta_top}")

    else:
       
        st.warning("⚠️ No se encontraron frutas confiables para clasificar.")

st.markdown('</div>', unsafe_allow_html=True)
