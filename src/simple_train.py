import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
import numpy as np

# Set up GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32

# Create a simple synthetic dataset (for demonstration)
print("Creating synthetic dataset...")
def create_synthetic_data(num_samples=1000, img_size=IMG_SIZE, num_classes=2):
    # Generate random images
    images = np.random.rand(num_samples, img_size, img_size, 3)
    # Generate random labels
    labels = np.random.randint(0, num_classes, num_samples)
    return images, labels

train_images, train_labels = create_synthetic_data(800)
val_images, val_labels = create_synthetic_data(200)

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)

val_ds = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_ds = val_ds.batch(BATCH_SIZE)

# Create a simple CNN model
print("Building model...")
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2, activation='softmax')
])

# Compile model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
print("Training model...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)

# Save model in the recommended format
model.save('models/plant_disease_model.keras')
print("Model saved to models/plant_disease_model.keras")

# Also save in HDF5 format for compatibility
model.save('models/plant_disease_model.h5')
print("Model also saved as HDF5 format")
