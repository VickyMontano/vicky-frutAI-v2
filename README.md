# === /prod/README.md ===

#  FRUT-AI - Clasificador de Frutas con SAM y EfficientNet

Este proyecto permite identificar **múltiples frutas** en una misma imagen utilizando:

- Segmentación automática con [SAM (Segment Anything Model)](https://segment-anything.com/) de Meta AI.
- Clasificación con una red neuronal basada en **EfficientNet-B3** entrenada en PyTorch.

---

##  Estructura del repositorio

```
frut-ai/
├── data/           # Datasets (opcional)
│   └── (archivos CSV o imágenes)
├── dev/
│   └── model_dev.ipynb     # Notebook de entrenamiento
├── prod/
│   ├── app.py              # Interfaz Streamlit
│   ├── utils.py            # Funciones auxiliares
│   ├── modelo.pth          # Modelo entrenado (PyTorch)
│   ├── requirements.txt    # Dependencias
│   └── README.md           # Este archivo
```

---

##  Cómo ejecutar la app localmente

### 1. Clonar el repositorio

```bash
git clone https://github.com/usuario/frut-ai.git
cd frut-ai/prod
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv
source venv/bin/activate       # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> Asegurate de tener `git` instalado para clonar el repo de SAM si usás requirements.txt con el repo remoto.

---

### 3. Ejecutar la app

```bash
streamlit run app.py
```

---

##  ¿Qué hace?

1. Te permite **subir una imagen con varias frutas**.
2. Aplica segmentación automática con SAM para encontrar objetos/frutas.
3. Clasifica cada fruta detectada usando EfficientNet-B3 entrenada.
4. Muestra una imagen con **bounding boxes** y etiquetas con nivel de confianza.

---

##  Modelo entrenado

- Arquitectura: EfficientNet-B3
- Dataset: 10 clases de frutas
- Salida: una predicción por fruta segmentada (filtrada por clase)
- Segmentador: `sam_vit_h_4b8939.pth` descargado desde Meta

---

##  Requisitos

- Python 3.8+
- GPU (opcional, pero acelera la segmentación SAM)
- Internet (si `segment-anything` se instala desde GitHub)

---

##  Créditos

- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- Proyecto académico para la cátedra de Redes Neuronales Profundas – UTN