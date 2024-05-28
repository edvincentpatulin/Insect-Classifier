from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import base64
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Ensure the temp directory exists
os.makedirs('temp', exist_ok=True)

# Load the trained model
model_path = '../resources/dataset/trained_model.h5'
class_names_path = '../resources/dataset/Classname.txt'

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading the model: {e}")
    exit()

# Load class names
try:
    with open(class_names_path, 'r') as f:
        class_names = f.read().splitlines()
    print("Class names loaded successfully.")
except Exception as e:
    print(f"Error loading class names: {e}")
    exit()

# Function to preprocess the image and make a prediction
def predict_image(model, img_path, class_names):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    
    # Remove the index/number from the predicted class name
    predicted_class_name = ' '.join(predicted_class.split(' ')[1:])
    
    return predicted_class_name

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file to a temporary location
    file_path = os.path.join('temp', file.filename)
    file.save(file_path)

    try:
        # Make prediction
        predicted_class = predict_image(model, file_path, class_names)

        # Encode the image to base64
        with open(file_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        return jsonify({'class': predicted_class, 'image': img_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the saved file
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == '__main__':
    app.run(debug=True)
