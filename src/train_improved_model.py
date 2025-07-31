import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import matplotlib.pyplot as plt
import numpy as np
import os
import requests
import zipfile

# Download real plant dataset
def download_dataset():
    urls = [
        "https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Tomato___healthy.zip",
        "https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Tomato___Bacterial_spot.zip",
        "https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Tomato___Early_blight.zip",
        "https://github.com/spMohanty/PlantVillage-Dataset/raw/master/raw/color/Tomato___Late_blight.zip"
    ]
    
    os.makedirs("data/tomato", exist_ok=True)
    
    for url in urls:
        filename = url.split("/")[-1]
        filepath = f"data/{filename}"
        
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            response = requests.get(url)
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            print(f"Extracting {filename}...")
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall("data/tomato/")
            os.remove(filepath)

# Download dataset
download_dataset()

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
DATA_DIR = "data/tomato"

# Create dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE)

class_names = train_ds.class_names
print("Classes:", class_names)

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Create improved model
def create_model():
    # Use pre-trained EfficientNetB0
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet')
    
    base_model.trainable = False  # Freeze base model
    
    # Add custom head with regularization
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        layers.Rescaling(1./255),
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
        layers.Dropout(0.3),
        layers.Dense(len(class_names), activation='softmax')
    ])
    
    return model

model = create_model()

# Compile with class weights to handle imbalance
class_counts = [len(os.listdir(f"data/tomato/{name}")) for name in class_names]
total = sum(class_counts)
class_weights = {i: total/(len(class_names)*count) for i, count in enumerate(class_counts)}
print("Class weights:", class_weights)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3,
    min_lr=0.00001)

# Train model
print("\nStarting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=20,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

# Save model
model.save('models/plant_disease_model_improved.h5')
print("\nModel saved to models/plant_disease_model_improved.h5")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.legend()

plt.savefig('training_history_improved.png')
plt.show()
