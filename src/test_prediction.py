import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Load model
try:
    model = tf.keras.models.load_model('models/plant_disease_model.h5')
except:
    model = tf.keras.models.load_model('models/plant_disease_model.keras')

# Create a synthetic image
print("Creating synthetic test image...")
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
img = Image.fromarray(img_array)
img.save("synthetic_test.jpg")

# Preprocess
img_array = tf.keras.utils.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)
img_array = img_array / 255.0

# Predict
predictions = model.predict(img_array)
score = tf.nn.softmax(predictions[0])
predicted_class = ['diseased', 'healthy'][np.argmax(score)]
confidence = np.max(score) * 100

print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

# Display
plt.imshow(img)
plt.title(f"{predicted_class} ({confidence:.2f}%)")
plt.axis('off')
plt.show()
