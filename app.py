import os
import numpy as np
import cv2
import tensorflow as tf
import streamlit as st
from sklearn.model_selection import train_test_split
from PIL import Image

# -------------------------------
# 1. Load and Preprocess Dataset
# -------------------------------
DATA_DIR = r"D:\CodeBase\Project\Image Processing\data"  # expects subfolders: buffalo, cattle
IMG_SIZE = (128, 128)

@st.cache_resource
def load_and_train_model():
    images = []
    labels = []
    class_names = sorted(os.listdir(DATA_DIR))  # Ensure consistent order
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = os.path.join(DATA_DIR, class_name)
        if not os.path.isdir(class_folder):
            continue
        for img_file in os.listdir(class_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(class_folder, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, IMG_SIZE)
                images.append(img)
                labels.append(class_to_idx[class_name])

    images = np.array(images, dtype=np.float32) / 255.0
    labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))

    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, stratify=labels, random_state=42
    )

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        verbose=0  # Hide training logs
    )

    return model, class_names

# Load model + class names only once
model, class_names = load_and_train_model()

# -------------------------------
# 2. Streamlit UI
# -------------------------------
st.title("🐄 Animal Classifier (CNN + OpenCV)")
st.write("Upload an image of an animal, and the model will predict its class.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and preprocess image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    img = img.convert("RGB")
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{pred_class}** ({confidence:.2f}% confidence)")