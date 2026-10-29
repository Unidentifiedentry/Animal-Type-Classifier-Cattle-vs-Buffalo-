import os
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. Setup data generator
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = train_datagen.flow_from_directory(
    "D:\\CodeBase\\Project\\Image Processing\\data",
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "D:\\CodeBase\\Project\\Image Processing\\data",
    target_size=(128, 128),
    batch_size=16,
    class_mode="categorical",
    subset="validation"
)

print("Class indices:", train_data.class_indices)

# 2. Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')  # cattle vs buffalo
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 3. Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=10
)

# 4. Plot accuracy
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.show()

# 5. Test model on a random image
class_names = list(train_data.class_indices.keys())
test_folder = random.choice(class_names)
test_img = random.choice(os.listdir(f"D:\\CodeBase\\Project\\Image Processing\\data\\buffalo"))

img = tf.keras.preprocessing.image.load_img(f"D:\\CodeBase\\Project\\Image Processing\\data\\buffalo\\{test_img}", target_size=(128,128))
img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

prediction = model.predict(img_array)
pred_class = class_names[np.argmax(prediction)]

plt.imshow(tf.keras.preprocessing.image.load_img(f"D:\\CodeBase\\Project\\Image Processing\\data\\cattle\\24.-Gangatiri_Indian-Cow-Breed.jpeg"))
plt.title(f"Predicted: {pred_class}")
plt.show()