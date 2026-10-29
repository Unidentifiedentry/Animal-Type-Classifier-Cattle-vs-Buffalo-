import os
import random
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. Load dataset manually using OpenCV
DATA_DIR = r"D:\CodeBase\Project\Image Processing\data"  # expects subfolders: buffalo, cattle
IMG_SIZE = (128, 128)

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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB
            img = cv2.resize(img, IMG_SIZE)
            images.append(img)
            labels.append(class_to_idx[class_name])

images = np.array(images, dtype=np.float32) / 255.0  # normalize
labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))

print(f"Loaded {len(images)} images across {len(class_names)} classes: {class_names}")

# 2. Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    images, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_val.shape[0]}")

# 3. Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),  # helps avoid overfitting
    tf.keras.layers.Dense(len(class_names), activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=10,
    batch_size=16
)

# 5. Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

# 6. Test on a random image
haar_cascade_path = "dog_cascade.xml"  # <-- Change to your cascade file
harrcascade = cv2.CascadeClassifier(haar_cascade_path)

# Map class index to name (like your `animal` dictionary)
animal = {i: name for i, name in enumerate(class_names)}

test_folder = random.choice(class_names)
test_img = random.choice([
    f for f in os.listdir(os.path.join(DATA_DIR, test_folder))
    if f.lower().endswith(('.jpg', '.jpeg', '.png'))
])
test_path = os.path.join(DATA_DIR, test_folder, test_img)

img = cv2.imread(test_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces_rect = harrcascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

for (x, y, w, h) in faces_rect:
    # Crop ROI and prepare for model
    faces_roi = img[y:y+h, x:x+w]
    faces_roi_rgb = cv2.cvtColor(faces_roi, cv2.COLOR_BGR2RGB)
    faces_roi_resized = cv2.resize(faces_roi_rgb, (128, 128))
    input_array = np.expand_dims(faces_roi_resized / 255.0, axis=0)

    # Predict class
    prediction = model.predict(input_array)
    label = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    # Draw rectangle + label text
    cv2.putText(img, f"{animal[label]} ({confidence:.1f}%)", (x, y - 10),
               cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 200), thickness=2)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

# Show result using OpenCV window
cv2.imshow("Detected & Classified Animals", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
