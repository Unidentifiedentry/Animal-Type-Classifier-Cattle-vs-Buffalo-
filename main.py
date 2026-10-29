''' 
requirements:
pip install opencv-python scikit-learn matplotlib joblib
'''
import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from train_model import train_classifier
from classify_image import classify_image

# === CONFIG ===
DATA_ROOT = "D:/CodeBase/Project/Image Processing/data"  # expects subfolders: buffalo, cattle
IMG_SIZE = (128, 128)
MODEL_PATHS = {'buffalo': 'buffalo_classifier.joblib', 'cattle': 'cattle_classifier.joblib'}
CLASS_NAMES = ['buffalo', 'cattle']

# === FEATURE EXTRACTION ===
def extract_features(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    # Example: Simple horn pattern detection (edge count)
    edges = cv2.Canny(gray, 100, 200)
    horn_pattern_score = np.sum(edges) / (height * width)
    return {
        'height': height,
        'width': width,
        'horn_pattern_score': horn_pattern_score
    }

# === LOAD IMAGES & LABELS ===
def load_images_and_labels(root_dir):
    images = {cls: [] for cls in CLASS_NAMES}
    labels = {cls: [] for cls in CLASS_NAMES}
    img_paths = {cls: [] for cls in CLASS_NAMES}
    debug_counts = {}
    for idx, class_name in enumerate(CLASS_NAMES):
        class_folder = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_folder):
            continue
        count = 0
        # Load images from subfolders and directly inside class_folder
        for root, dirs, files in os.walk(class_folder):
            for img_file in files:
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img_path = os.path.join(root, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                img = cv2.resize(img, IMG_SIZE)
                # Store both color and grayscale
                images[class_name].append(img)
                labels[class_name].append(idx)
                img_paths[class_name].append(img_path)
                count += 1
        debug_counts[class_name] = count
    print(f"Image counts per class: {debug_counts}")
    return images, labels, img_paths

# === MAIN ===
def main():
    # Load images and labels
    images, labels, img_paths = load_images_and_labels(DATA_ROOT)
    # Train and save a separate model for each class (one-vs-rest)
    for class_name in CLASS_NAMES:
        X = []
        y = []
        # One-vs-rest: current class is 1, others are 0
        for cls in CLASS_NAMES:
            for img in images[cls]:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                X.append(gray.flatten())
                y.append(1 if cls == class_name else 0)
        X = np.array(X)
        y = np.array(y)
        if len(set(y)) < 2:
            print(f"Not enough data for {class_name} classifier. Skipping.")
            continue
        print(f"Training {class_name} classifier with {len(X)} samples...")
        clf = train_classifier(X, y, model_path=MODEL_PATHS[class_name])
    # Show a random color image and its predicted class for both models
    all_img_paths = sum([img_paths[cls] for cls in CLASS_NAMES], [])
    all_images = sum([images[cls] for cls in CLASS_NAMES], [])
    if not all_img_paths:
        print("No images found. Exiting.")
        return
    idx = np.random.choice(len(all_img_paths))
    img_path = all_img_paths[idx]
    img = all_images[idx]
    print(f"\nClassifying image: {img_path}")
    results = {}
    for class_name in CLASS_NAMES:
        clf = joblib.load(MODEL_PATHS[class_name])
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        pred = clf.predict(gray.flatten().reshape(1, -1))[0]
        results[class_name] = pred
        print(f"Predicted by {class_name} model: {['not '+class_name, class_name][pred]}")
    features = extract_features(img_path)
    print(f"Extracted features: {features}")
    # Show color image
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {', '.join([f'{cls}: {['not '+cls, cls][results[cls]]}' for cls in CLASS_NAMES])}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()