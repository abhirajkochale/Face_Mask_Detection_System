# 😷 Face Mask Detection System

## 📌 Objective
Detect whether individuals are wearing masks in real-time using a webcam.

## 🛠 Tech Stack
- OpenCV
- TensorFlow / Keras (CNN)
- Haar Cascade Classifier
- Streamlit (for UI)

## 📖 Description
This system captures live video feed from a webcam and detects faces using OpenCV’s Haar cascade.  
A trained Convolutional Neural Network (CNN) classifies whether the person is **wearing a mask** or **not wearing a mask**.  

- Green Bounding Box ✅ → With Mask  
- Red Bounding Box ❌ → Without Mask  

## 🚀 Features
- Real-time face mask detection via webcam
- Trained CNN on labeled dataset (Mask / No Mask)
- Live feedback with bounding boxes
- Optional: Alert notification if no mask detected
- Streamlit app for interactive use

## 📦 Dataset
This project uses the [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) from Kaggle.  
Download and extract it into the `data_raw/` folder before running the training script.
