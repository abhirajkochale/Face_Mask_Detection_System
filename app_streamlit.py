import json
from pathlib import Path
import numpy as np
import cv2
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

st.set_page_config(page_title="Face Mask Detection", page_icon="😷", layout="centered")

st.title("😷 Face Mask Detection (Image Upload)")
st.write("Upload an image. The app will detect faces and predict **with_mask** or **without_mask** for each face.")

@st.cache_resource
def load_assets():
    model = load_model('models/mask_detector.h5')
    idx_to_class = json.loads(Path('models/label_map.json').read_text())
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    return model, idx_to_class, face_cascade

model, idx_to_class, face_cascade = load_assets()

uploaded = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read the image.")
    else:
        display = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
        for (x,y,w,h) in faces:
            face = img[y:y+h, x:x+w]
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face_resized = cv2.resize(face_rgb, (224,224)).astype('float32')
            face_pre = preprocess_input(face_resized)
            probs = model.predict(np.expand_dims(face_pre, axis=0), verbose=0)[0]
            idx = int(np.argmax(probs))
            label = idx_to_class.get(idx, str(idx))
            conf = float(np.max(probs))
            color = (0,255,0) if 'with' in label else (0,0,255)
            cv2.rectangle(display, (x,y), (x+w,y+h), color, 2)
            cv2.putText(display, f"{label}: {conf*100:.1f}%", (x, y-10 if y-10>20 else y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), caption="Predictions", use_column_width=True)
        st.success("Done!")
else:
    st.info("Upload an image to get started.")
