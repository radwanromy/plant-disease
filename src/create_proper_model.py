import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

print("Creating a properly compiled model...")

# Create a simple model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1./255),
    layers.Conv2D(16, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(2, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Create some dummy data and train briefly (to ensure compilation is saved)
print("Training briefly to save compilation info...")
dummy_data = np.random.rand(10, 224, 224, 3)
dummy_labels = np.random.randint(0, 2, 10)
model.fit(dummy_data, dummy_labels, epochs=1, verbose=0)

# Save the model with compilation info
model.save('models/plant_disease_model_compiled.h5')
print("Model saved with compilation info: models/plant_disease_model_compiled.h5")

# Also save in Keras format
model.save('models/plant_disease_model_compiled.keras')
print("Model also saved in Keras format: models/plant_disease_model_compiled.keras")
