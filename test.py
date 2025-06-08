import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image # For direct image handling from uploads
import io
import json # For loading class indices

# --- Machine Learning Model Imports ---
import tensorflow as tf
from tensorflow.keras.models import load_model
# import pickle # No longer needed if health prediction is also image-based

# --- Configuration ---
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads' # Where uploaded images will be temporarily stored
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Ensure the upload and models folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- Global variables for models and class labels ---
cow_id_model = None
cow_class_labels = None
IMG_HEIGHT_COW_ID, IMG_WIDTH_COW_ID = 150, 150 # Image dimensions for cow_id_model

health_eye_model = None # Renamed for clarity: this is the conjunctivitis detection model
health_eye_class_labels = None
IMG_HEIGHT_EYE, IMG_WIDTH_EYE = 224, 224 # Image dimensions for health_eye_model
CONJUNCTIVITIS_THRESHOLD = 0.545 # Your specific threshold from Model 2

# --- Load Pre-trained Machine Learning Models ---
@app.before_request
def load_models_once():
    """Load models only once when the first request comes in."""
    global cow_id_model, cow_class_labels, health_eye_model, health_eye_class_labels

    # Load Cow Identification Model
    if cow_id_model is None:
        try:
            cow_id_model_path = os.path.join('models', 'cow_id.h5')
            if os.path.exists(cow_id_model_path):
                cow_id_model = load_model(cow_id_model_path)
                print(f"Loaded Cow ID Model from {cow_id_model_path}")
            else:
                print(f"Warning: Cow ID model not found at {cow_id_model_path}")

            cow_class_indices_path = os.path.join('models', 'cow_class_indices.json')
            if os.path.exists(cow_class_indices_path):
                with open(cow_class_indices_path, 'r') as f:
                    loaded_class_indices = json.load(f)
                cow_class_labels = [None] * len(loaded_class_indices)
                for label, index in loaded_class_indices.items():
                    cow_class_labels[index] = label
                print(f"Loaded cow class labels: {cow_class_labels}")
            else:
                print(f"Warning: Cow class indices not found at {cow_class_indices_path}. Cow ID results may be incorrect.")
                cow_class_labels = []

        except Exception as e:
            print(f"Error loading Cow ID model or labels: {e}")
            cow_id_model = None
            cow_class_labels = None

    # Load Health Eye Model (Conjunctivitis Detection)
    if health_eye_model is None:
        try:
            health_eye_model_path = os.path.join('models', 'cow_eye.h5')
            if os.path.exists(health_eye_model_path):
                health_eye_model = load_model(health_eye_model_path)
                print(f"Loaded Health Eye Model from {health_eye_model_path}")
            else:
                print(f"Warning: Health Eye model not found at {health_eye_model_path}")
            
            health_class_indices_path = os.path.join('models', 'conjunctivitis_class_indices.json')
            if os.path.exists(health_class_indices_path):
                with open(health_class_indices_path, 'r') as f:
                    loaded_health_class_indices = json.load(f)
                health_eye_class_labels = [None] * len(loaded_health_class_indices)
                for label, index in loaded_health_class_indices.items():
                    health_eye_class_labels[index] = label
                print(f"Loaded health eye class labels: {health_eye_class_labels}")
            else:
                print(f"Warning: Conjunctivitis class indices not found at {health_class_indices_path}. Health results may be incorrect.")
                # Default labels if file is missing (assuming 'healthy' is index 0, 'conjunctivitis' is index 1)
                health_eye_class_labels = ['healthy', 'conjunctivitis']

        except Exception as e:
            print(f"Error loading Health Eye model or labels: {e}")
            health_eye_model = None
            health_eye_class_labels = None

# --- Helper Functions ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(file_content, target_height, target_width):
    """
    Preprocesses an image from its binary content for a Keras model.
    Ensures image is resized, converted to RGB, and normalized.
    """
    img = Image.open(io.BytesIO(file_content))
    img = img.resize((target_width, target_height))

    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0 # Normalize pixel values
    return img_array

# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/identify_cow', methods=['POST'])
def identify_cow():
    if cow_id_model is None or cow_class_labels is None:
        return jsonify({'error': 'Cow identification service not ready. Models not loaded.'}), 503

    if 'cow_image' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['cow_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Read file content once to save and process
        file_content = file.read() 
        
        # Save the uploaded file for display on the frontend
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(file_content)

        try:
            processed_image = preprocess_image(file_content, IMG_HEIGHT_COW_ID, IMG_WIDTH_COW_ID)
            
            predictions = cow_id_model.predict(processed_image)
            predicted_index = np.argmax(predictions[0])
            
            predicted_label = "Unknown Cow"
            if predicted_index < len(cow_class_labels):
                predicted_label = cow_class_labels[predicted_index]
            else:
                print(f"Warning: Predicted index {predicted_index} is out of bounds for cow_class_labels ({len(cow_class_labels)}).")

            confidence = np.max(predictions[0]) * 100

            return jsonify({
                'success': True,
                'message': 'Cow identified successfully!',
                'cow_id': predicted_label,
                'confidence': f"{confidence:.2f}%",
                'image_url': f'/{filepath}'
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath) # Clean up uploaded file on error
            return jsonify({'error': f'Error processing image for identification: {e}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/predict_health', methods=['POST'])
def predict_health():
    # This route now handles eye image upload for conjunctivitis detection
    if health_eye_model is None or health_eye_class_labels is None:
        return jsonify({'error': 'Health prediction service not ready. Models not loaded.'}), 503

    if 'eye_image' not in request.files: # Frontend will send 'eye_image'
        return jsonify({'error': 'No eye image part'}), 400
    
    file = request.files['eye_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        file_content = file.read()
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(file_content)

        try:
            processed_image = preprocess_image(file_content, IMG_HEIGHT_EYE, IMG_WIDTH_EYE)
            
            prediction = health_eye_model.predict(processed_image)[0][0] # Get the scalar prediction
            
            # Determine status based on prediction and the threshold from your model training
            health_status = ""
            if health_eye_class_labels and len(health_eye_class_labels) == 2:
                # Assuming index 0 is 'healthy' and index 1 is 'conjunctivitis' or vice-versa
                # You must confirm this mapping from your 'conjunctivitis_class_indices.json'
                if health_eye_class_labels[0] == 'healthy': # If 'healthy' is index 0
                    if prediction < CONJUNCTIVITIS_THRESHOLD:
                        health_status = health_eye_class_labels[0] # Healthy
                    else:
                        health_status = health_eye_class_labels[1] # Conjunctivitis
                else: # If 'healthy' is index 1 (less common)
                    if prediction < CONJUNCTIVITIS_THRESHOLD:
                        health_status = health_eye_class_labels[0] # Conjunctivitis
                    else:
                        health_status = health_eye_class_labels[1] # Healthy
            else: # Fallback if labels are missing or unexpected
                health_status = "Conjunctivitis Detected" if prediction >= CONJUNCTIVITIS_THRESHOLD else "Healthy Eye"

            return jsonify({
                'success': True,
                'message': 'Health status predicted successfully!',
                'status': health_status,
                'prediction_score': f"{prediction:.4f}",
                'image_url': f'/{filepath}' # Show the uploaded eye image
            })
        except Exception as e:
            if os.path.exists(filepath):
                os.remove(filepath) # Clean up uploaded file on error
            return jsonify({'error': f'Error processing image for health prediction: {e}'}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    with app.app_context():
        load_models_once() # Ensure models are loaded when starting directly
    app.run(debug=True)