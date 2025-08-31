import os, json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4

DATA_DIR = Path('data')
TRAIN_DIR = DATA_DIR/'train'
VAL_DIR = DATA_DIR/'val'

MODELS_DIR = Path('models')
REPORTS_DIR = Path('reports')
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Data generators
train_aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
)

val_aug = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_aug.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_gen = val_aug.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_gen.class_indices)
print("[INFO] class_indices:", train_gen.class_indices)

# Save label map
label_map_path = MODELS_DIR/'label_map.json'
with open(label_map_path, 'w') as f:
    json.dump({v:k for k,v in train_gen.class_indices.items()}, f, indent=2)
print(f"[INFO] Saved label map to {label_map_path}")

# Build model (Transfer Learning)
base = MobileNetV2(weights='imagenet', include_top=False, input_tensor=Input(shape=(IMG_SIZE, IMG_SIZE, 3)))
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=outputs)

# Freeze base for initial training
for layer in base.layers:
    layer.trainable = False

opt = Adam(learning_rate=LR)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print(model.summary())

H = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# Unfreeze some top layers for fine-tuning (optional)
for layer in base.layers[-20:]:
    layer.trainable = True
opt_ft = Adam(learning_rate=LR/10)
model.compile(loss='categorical_crossentropy', optimizer=opt_ft, metrics=['accuracy'])
H_ft = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=3
)

# Evaluate
val_gen.reset()
pred_probs = model.predict(val_gen)
y_pred = np.argmax(pred_probs, axis=1)
y_true = val_gen.classes

target_names = [k for k,_ in sorted(train_gen.class_indices.items(), key=lambda x: x[1])]
report = classification_report(y_true, y_pred, target_names=target_names, digits=4)
print(report)
with open(REPORTS_DIR/'val_classification_report.txt','w') as f:
    f.write(report)

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("[INFO] Confusion matrix:\n", cm)

# Plot training
def plot_history(H_list, labels):
    # Combine histories
    acc = []
    val_acc = []
    loss = []
    val_loss = []
    for H in H_list:
        acc += H.history['accuracy']
        val_acc += H.history['val_accuracy']
        loss += H.history['loss']
        val_loss += H.history['val_loss']
    # Plot
    plt.figure()
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training/Validation Accuracy')
    plt.savefig(REPORTS_DIR/'training_accuracy.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training/Validation Loss')
    plt.savefig(REPORTS_DIR/'training_loss.png', bbox_inches='tight')
    plt.close()

plot_history([H, H_ft], ['initial','finetune'])

# Save model
model_path = MODELS_DIR/'mask_detector.h5'
model.save(model_path)
print(f"[OK] Saved model to {model_path}")
print(f"[OK] Reports saved to {REPORTS_DIR}")
