import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load model
try:
    model = tf.keras.models.load_model('models/plant_disease_model_compiled.h5')
except:
    model = tf.keras.models.load_model('models/plant_disease_model_compiled.keras')

# Create test image
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)

# Preprocess
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
class_names = ['diseased', 'healthy']
predicted_class = class_names[np.argmax(score)]
confidence = np.max(score) * 100

print(f"Quick test successful!")
print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
