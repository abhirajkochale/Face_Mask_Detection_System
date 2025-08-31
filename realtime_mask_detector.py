import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMG_SIZE = 224

model = load_model('models/mask_detector.h5')
with open('models/label_map.json','r') as f:
    idx_to_class = json.load(f)

# Reverse map to ensure numeric keys are strings
idx_to_class = {int(k): v for k, v in idx_to_class.items()}

cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)
if face_cascade.empty():
    raise RuntimeError(f'Could not load Haar cascade from {cascade_path}')

cap = cv2.VideoCapture(0)

print("[INFO] Press 'q' to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5,
                                          minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (IMG_SIZE, IMG_SIZE)).astype('float32')
        face_pre = preprocess_input(face_resized)
        face_input = np.expand_dims(face_pre, axis=0)

        probs = model.predict(face_input, verbose=0)[0]
        idx = int(np.argmax(probs))
        label = idx_to_class.get(idx, str(idx))
        conf = float(np.max(probs))

        color = (0, 255, 0) if 'with' in label else (0, 0, 255)
        text = f"{label}: {conf*100:.1f}%"
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        y_text = y - 10 if y - 10 > 20 else y + 20
        cv2.putText(frame, text, (x, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Face Mask Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
