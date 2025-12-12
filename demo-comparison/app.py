"""
Pneumonia Detection Model Comparison Demo
Compares Centralized vs Federated Learning Models
"""
import os
import io
import base64
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from huggingface_hub import from_pretrained_keras
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global model variables
centralized_model = None
federated_model = None


def load_models():
    """Load both centralized and federated models"""
    global centralized_model, federated_model
    
    try:
        # Load centralized model from HuggingFace
        logger.info("Loading centralized model from HuggingFace...")
        centralized_model = from_pretrained_keras("ryefoxlime/PneumoniaDetection")
        logger.info("✓ Centralized model loaded successfully")
        
        # For demo purposes, use the same model for FL (in production, this would be your FL-trained model)
        logger.info("Loading federated learning model...")
        federated_model = from_pretrained_keras("ryefoxlime/PneumoniaDetection")
        logger.info("✓ Federated model loaded successfully (using same model for demo)")
        
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False


def preprocess_image(image: Image.Image, target_size=(224, 224)):
    """
    Preprocess image for model prediction
    
    Args:
        image: PIL Image
        target_size: Target size for resizing
        
    Returns:
        Preprocessed numpy array ready for prediction
    """
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize
    image = image.resize(target_size)
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array.astype('float32') / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array


def predict_with_model(model, image_array):
    """
    Make prediction with given model
    
    Args:
        model: Keras model
        image_array: Preprocessed image array
        
    Returns:
        Dictionary with prediction results
    """
    try:
        prediction = model.predict(image_array, verbose=0)
        
        # Extract probability (assuming binary classification)
        if prediction.shape[-1] == 1:
            pneumonia_prob = float(prediction[0][0])
        else:
            pneumonia_prob = float(prediction[0][1])
        
        normal_prob = 1.0 - pneumonia_prob
        
        # Determine class
        predicted_class = "Pneumonia" if pneumonia_prob > 0.5 else "Normal"
        confidence = max(pneumonia_prob, normal_prob)
        
        return {
            "class": predicted_class,
            "confidence": float(confidence * 100),
            "pneumonia_probability": float(pneumonia_prob * 100),
            "normal_probability": float(normal_prob * 100)
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return {
            "error": str(e)
        }


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    try:
        # Read and process image
        image = Image.open(io.BytesIO(file.read()))
        
        # Preprocess for both models
        processed_image = preprocess_image(image)
        
        # Get predictions from both models
        centralized_result = predict_with_model(centralized_model, processed_image)
        federated_result = predict_with_model(federated_model, processed_image)
        
        # Convert image to base64 for display
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        
        return jsonify({
            "success": True,
            "image": img_str,
            "centralized": centralized_result,
            "federated": federated_result
        })
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    models_loaded = centralized_model is not None and federated_model is not None
    return jsonify({
        "status": "healthy" if models_loaded else "loading",
        "models_loaded": models_loaded
    })


if __name__ == '__main__':
    print("=" * 80)
    print("🏥 PNEUMONIA DETECTION MODEL COMPARISON DEMO")
    print("=" * 80)
    print("\n📦 Loading models...")
    
    if load_models():
        print("\n✅ Models loaded successfully!")
        print("\n🌐 Starting Flask server...")
        print("   Access the demo at: http://localhost:5000")
        print("\n" + "=" * 80)
        app.run(debug=True, host='0.0.0.0', port=7500)
    else:
        print("\n❌ Failed to load models. Please check the error messages above.")
        print("   Make sure you have installed: pip install huggingface_hub tf_keras")
