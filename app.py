from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

app = Flask(__name__)

# Load the model
try:
    model = tf.keras.models.load_model('models/plant_disease_model.h5')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print("Model loaded successfully")
except:
    try:
        model = tf.keras.models.load_model('models/plant_disease_model.keras')
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model loaded successfully (.keras format)")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

# Class names
class_names = ['diseased', 'healthy', 'bacterial_spot', 'early_blight', 'late_blight',
               'leaf_mold', 'septoria_leaf_spot', 'spider_mites', 'target_spot',
               'mosaic_virus', 'yellow_leaf_curl_virus']

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        # Read and process the image
        image_bytes = file.read()
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img = img.resize((224, 224))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = img_array / 255.0
        
        # Make prediction
        if model:
            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = np.max(score) * 100
            
            # Save image for display
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img.save(filepath)
            
            return jsonify({
                'prediction': predicted_class,
                'confidence': f"{confidence:.2f}%",
                'image_url': f"/static/uploads/{filename}"
            })
        else:
            return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data'}), 400
    
    # Decode base64 image
    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    
    # Process image
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = img_array / 255.0
    
    # Make prediction
    if model:
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = np.max(score) * 100
        
        return jsonify({
            'prediction': predicted_class,
            'confidence': f"{confidence:.2f}%",
            'image_data': data['image']  # Return original image for display
        })
    else:
        return jsonify({'error': 'Model not loaded'}), 500

def allowed_file(filename):
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # Changed to port 5001
