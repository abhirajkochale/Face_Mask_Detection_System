# Face Mask Detection – Starter Kit

This is a minimal, **beginner-friendly** implementation of a Face Mask Detection system.

## 0) Project Structure

```
face-mask-detection/
├─ data_raw/                       # Put your raw dataset here (two folders: with_mask, without_mask)
├─ data/
│  ├─ train/                       # auto-filled by split_data.py
│  ├─ val/
│  └─ test/
├─ models/
│  ├─ mask_detector.h5             # saved after training
│  └─ label_map.json
├─ reports/
│  ├─ training_accuracy.png
│  ├─ training_loss.png
│  └─ val_classification_report.txt
├─ split_data.py
├─ train_mask_detector.py
├─ realtime_mask_detector.py
├─ app_streamlit.py                # optional: image upload UI
└─ requirements.txt
```

## 1) Create environment & install requirements

```bash
# (Windows)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# (macOS/Linux)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> If TensorFlow install gives issues on macOS M1/M2, consider `tensorflow-macos` and `tensorflow-metal`.

## 2) Get a dataset

Download any **Face Mask** dataset (e.g., Kaggle). You need exactly two folders inside `data_raw/`:

```
data_raw/
├─ with_mask/
└─ without_mask/
```

Each must contain images of that class.

## 3) Split the dataset

```bash
python split_data.py
```

This fills `data/train`, `data/val`, and `data/test` with an 80/10/10 split.

## 4) Train the model

```bash
python train_mask_detector.py
```

Outputs:
- `models/mask_detector.h5`, `models/label_map.json`
- Plots in `reports/` and a classification report.

## 5) Run real-time detection (webcam)

```bash
python realtime_mask_detector.py
```

Press `q` to quit.

## 6) (Optional) Simple web app

```bash
streamlit run app_streamlit.py
```

Upload an image to see predictions.

## Tips

- If webcam index 0 doesn’t work, try `cv2.VideoCapture(1)` etc.
- Low accuracy? Add more training images or train a few more epochs.
- Lighting affects detection; try the OpenCV DNN face detector for robustness.

## License

For educational use.
