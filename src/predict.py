import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import sys

def main():
    # Try to load compiled model first
    model_path = None
    for path in ['models/plant_disease_model_compiled.h5', 
                 'models/plant_disease_model_compiled.keras',
                 'models/plant_disease_model.h5', 
                 'models/plant_disease_model.keras']:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("Error: No model found. Please run 'python src/create_proper_model.py' first.")
        return
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"Loaded model from: {model_path}")
        
        # The model should already be compiled, but just in case
        if not model.optimizer:
            model.compile(optimizer='adam',
                         loss='sparse_categorical_crossentropy',
                         metrics=['accuracy'])
            print("Model compiled (was missing optimizer)")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get image path from command line or use default
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = "sample_leaf.jpg"
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        print("Creating a synthetic test image instead...")
        
        # Create synthetic image
        img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save("synthetic_test.jpg")
        image_path = "synthetic_test.jpg"
        print(f"Created synthetic image: {image_path}")

    # Predict
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_names = ['diseased', 'healthy']
        predicted_class = class_names[np.argmax(score)]
        confidence = np.max(score) * 100

        print(f"Prediction: {predicted_class} ({confidence:.2f}%)")

        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.2f}%)")
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()
