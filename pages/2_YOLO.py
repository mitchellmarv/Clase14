from PIL import Image
import io
import json
import streamlit as st
import numpy as np
import pandas as pd
import torch
import base64
import paho.mqtt.client as paho


st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# ---------- Configuración MQTT (igual que en el ESP32 de Wokwi) ----------
BROKER = "broker.mqttdashboard.com"
PORT = 1883
# Distinto al ID del ESP32 ("Lizalda"): si fueran iguales, el broker
# desconectaría a uno de los dos cada vez que el otro entrara.
CLIENT_ID = "Lizalda_App"
TOPIC_GESTO = "lizalda/IMIA"

# Si se detecta este objeto se envía "Abre"; si no, "Cierra" (cámbialo a tu gusto)
OBJETO_ABRE = "person"


def crear_cliente():
    """Compatible con paho-mqtt 1.x y 2.x."""
    try:
        return paho.Client(paho.CallbackAPIVersion.VERSION1, CLIENT_ID)
    except AttributeError:
        return paho.Client(CLIENT_ID)


def publicar_gesto(comando):
    """Conecta, publica, espera confirmación y se desconecta."""
    try:
        client = crear_cliente()
        client.connect(BROKER, PORT, keepalive=30)
        client.loop_start()
        info = client.publish(TOPIC_GESTO, json.dumps({"gesto": comando}))
        info.wait_for_publish(timeout=5)
        client.loop_stop()
        client.disconnect()
        st.toast(f"Gesto enviado: {comando}")
    except Exception as e:
        st.error(f"No se pudo enviar el gesto: {e}")


@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None

st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.")

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()

if model:
    with st.sidebar:
        st.title("Parámetros")
        st.subheader("Configuración de detección")
        conf_threshold = st.slider("Confianza mínima", 0.0, 1.0, 0.25, 0.01)
        iou_threshold  = st.slider("Umbral IoU", 0.0, 1.0, 0.45, 0.01)
        max_det        = st.number_input("Detecciones máximas", 10, 2000, 1000, 10)

    picture = st.camera_input("Capturar imagen", key="camera")

    if picture:
        bytes_data = picture.getvalue()

        # Decodificar con Pillow en lugar de cv2 (evita dependencia libGL)
        #pil_img  = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        #np_img   = np.array(pil_img)   # array RGB

        pil_img = Image.open(io.BytesIO(bytes_data)).convert("RGB")
        np_img  = np.array(pil_img)[..., ::-1]  # RGB → BGR para que YOLO procese bien

        
        with st.spinner("Detectando objetos..."):
            try:
                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )
            except Exception as e:
                st.error(f"Error durante la detección: {str(e)}")
                st.stop()

        result    = results[0]
        boxes     = result.boxes
        annotated = result.plot()              # devuelve BGR numpy array
        annotated_rgb = annotated[:, :, ::-1]  # BGR → RGB sin cv2

        # Enviar comando al ESP32 según lo detectado
        nombres_detectados = (
            {model.names[int(b.cls.item())] for b in boxes}
            if boxes is not None else set()
        )
        publicar_gesto("Abre" if OBJETO_ABRE in nombres_detectados else "Cierra")

        col1, col2 = st.columns(2)


        with col1:
            st.subheader("Imagen con detecciones")
            buf = io.BytesIO()
            Image.fromarray(annotated_rgb.copy()).save(buf, format="JPEG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            st.html(f'<img src="data:image/jpeg;base64,{b64}" style="width:100%">')

        with col2:
            st.subheader("Objetos detectados")
            if boxes is not None and len(boxes) > 0:
                label_names    = model.names
                category_count = {}
                category_conf  = {}

                for box in boxes:
                    cat  = int(box.cls.item())
                    conf = float(box.conf.item())
                    category_count[cat] = category_count.get(cat, 0) + 1
                    category_conf.setdefault(cat, []).append(conf)

                data = [
                    {
                        "Categoría":          label_names[cat],
                        "Cantidad":           count,
                        "Confianza promedio": f"{np.mean(category_conf[cat]):.2f}"
                    }
                    for cat, count in category_count.items()
                ]

                df = pd.DataFrame(data)
                st.dataframe(df, use_container_width=True)
                st.bar_chart(df.set_index("Categoría")["Cantidad"])
            else:
                st.info("No se detectaron objetos con los parámetros actuales.")
                st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
else:
    st.error("No se pudo cargar el modelo. Verifica las dependencias e inténtalo nuevamente.")
    st.stop()

st.markdown("---")
st.caption("**Acerca de la aplicación**: Detección de objetos con YOLOv5 + Streamlit + PyTorch.")
