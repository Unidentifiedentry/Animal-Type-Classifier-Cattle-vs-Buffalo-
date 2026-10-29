# requirements:
# pip install opencv-python scikit-learn matplotlib

import numpy as np
import joblib
import cv2
import os
import matplotlib.pyplot as plt

# === CONFIG ===
IMG_SIZE = (128, 128)
MODEL_PATH = "buffalo_classifier.joblib"

# === CLASSIFY IMAGE ===
def classify_image(img_path, class_names):
    clf = joblib.load(MODEL_PATH)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_flat = img.flatten().reshape(1, -1)
    pred = clf.predict(img_flat)[0]
    print(f"Predicted class: {class_names[pred]}")
    plt.imshow(img, cmap='gray')
    plt.title(f"Predicted: {class_names[pred]}")
    plt.axis('off')
    plt.show()
    return class_names[pred]

# Example usage:
# class_names = ['buffalo', 'cattle']
# classify_image('path/to/image.jpg', class_names)