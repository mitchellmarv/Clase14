import json

import streamlit as st
import cv2
import numpy as np
#from PIL import Image
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import paho.mqtt.client as paho

import platform

# ---------- Configuración MQTT (igual que en el ESP32 de Wokwi) ----------
BROKER = "broker.mqttdashboard.com"
PORT = 1883
# Distinto al ID del ESP32 ("Lizalda"): si fueran iguales, el broker
# desconectaría a uno de los dos cada vez que el otro entrara.
CLIENT_ID = "Lizalda_App"
TOPIC_GESTO = "lizalda/IMIA"

# Qué comando se envía según el gesto reconocido (cámbialo a tu gusto)
GESTO_A_COMANDO = {
    "Izquierda": "Abre",
    "Arriba": "Cierra",
}


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


# Muestra la versión de Python junto con detalles adicionales
st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("Reconocimiento de Imágenes")
#st.write("Versión de Python:", platform.python_version())
#image = Image.open('OIG5.jpg')
#st.image(image, width=350)

import base64

with open("OIG7.jpg", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()
st.html(f'<img src="data:image/jpeg;base64,{b64}" width="350">')


with st.sidebar:
    st.subheader("Usando un modelo entrenado en teachable Machine puedes Usarlo en esta app para identificar")
img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    # To read image file buffer with OpenCV:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
   #To read image file buffer as a PIL Image:
    img = Image.open(img_file_buffer)

    newsize = (224, 224)
    img = img.resize(newsize)
    # To convert PIL Image to numpy array:
    img_array = np.array(img)

    # Normalize the image
    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array

    # run the inference
    prediction = model.predict(data)
    print(prediction)
    if prediction[0][0]>0.5:
      st.header('Izquierda, con Probabilidad: '+str( prediction[0][0]) )
      publicar_gesto(GESTO_A_COMANDO["Izquierda"])
    if prediction[0][1]>0.5:
      st.header('Arriba, con Probabilidad: '+str( prediction[0][1]))
      publicar_gesto(GESTO_A_COMANDO["Arriba"])
    #if prediction[0][2]>0.5:
    # st.header('Derecha, con Probabilidad: '+str( prediction[0][2]))
