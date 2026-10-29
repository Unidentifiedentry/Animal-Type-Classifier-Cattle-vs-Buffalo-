# requirements:
# pip install opencv-python scikit-learn matplotlib

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# === CONFIG ===
DATA_ROOT = "D:/CodeBase/Project/Image Processing/data/buffalo/images"  # root folder containing BLFxxxx subfolders
IMG_SIZE = (128, 128)

# === LOAD IMAGES FROM ALL SUBFOLDERS ===
def load_images_from_subfolders(root_dir):
    images = []
    img_paths = []
    for subfolder in os.listdir(root_dir):
        subfolder_path = os.path.join(root_dir, subfolder)
        if not os.path.isdir(subfolder_path):
            continue
        for img_file in os.listdir(subfolder_path):
            img_path = os.path.join(subfolder_path, img_file)
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, IMG_SIZE)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img.flatten())
            img_paths.append(img_path)
    return np.array(images), img_paths

# === MAIN PIPELINE ===
X, img_paths = load_images_from_subfolders(DATA_ROOT)
print(f"Loaded {len(X)} images from all subfolders.")

if len(X) == 0:
    print("No images found. Please check your DATA_ROOT path and image files.")
else:
    # Example: Show random image
    def show_random_image():
        idx = np.random.choice(len(X))
        img = X[idx].reshape(IMG_SIZE)
        plt.imshow(img, cmap='gray')
        plt.title(f"Random Buffalo Image\n{os.path.basename(img_paths[idx])}")
        plt.axis('off')
        plt.show()

    # Example usage:
    show_random_image()

    # You can add clustering, anomaly detection, or other analysis here