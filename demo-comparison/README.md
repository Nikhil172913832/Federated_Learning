# 🏥 Pneumonia Detection: Model Comparison Demo

A modern, aesthetic web application to compare **Centralized Learning** vs **Federated Learning** approaches for pneumonia detection from chest X-ray images.

![Demo Banner](https://img.shields.io/badge/Demo-Pneumonia_Detection-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![Flask](https://img.shields.io/badge/Flask-3.0-red)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)

## 🌟 Features

- **Dual Model Comparison**: Side-by-side comparison of centralized and federated learning models
- **Modern UI/UX**: Beautiful, responsive design with gradient backgrounds and smooth animations
- **Drag & Drop Upload**: Easy image upload with drag-and-drop support
- **Real-time Predictions**: Instant analysis with visual probability breakdowns
- **Agreement Indicator**: Visual indicator showing model agreement/disagreement
- **Mobile Responsive**: Works seamlessly on desktop, tablet, and mobile devices

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- At least 4GB RAM (for model loading)
- Modern web browser (Chrome, Firefox, Safari, Edge)

## 🚀 Quick Start

### 1. Navigate to Demo Directory

```bash
cd demo-comparison
```

### 2. Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: The installation might take a few minutes as it downloads TensorFlow and other dependencies.

### 4. Run the Application

```bash
python app.py
```

You should see:

```
🏥 PNEUMONIA DETECTION MODEL COMPARISON DEMO
================================================================================

📦 Loading models...
✓ Centralized model loaded successfully
✓ Federated model loaded successfully (using same model for demo)

✅ Models loaded successfully!

🌐 Starting Flask server...
   Access the demo at: http://localhost:5000

================================================================================
```

### 5. Open in Browser

Navigate to: **http://localhost:5000**

## 📁 Project Structure

```
demo-comparison/
├── app.py                      # Flask backend application
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── models/                     # Directory for models (auto-created)
├── static/
│   ├── css/
│   │   └── style.css          # Modern styling
│   └── js/
│       └── main.js            # Frontend logic
└── templates/
    └── index.html             # Main HTML template
```

## 🎯 How to Use

1. **Launch the Application**
   - Start the Flask server as described above
   - Open your browser to http://localhost:5000

2. **Upload an Image**
   - Click the upload area or drag and drop a chest X-ray image
   - Supported formats: PNG, JPG, JPEG
   - Maximum file size: 16MB

3. **Analyze**
   - Click the "Analyze Image" button
   - Wait for the analysis (usually takes 2-5 seconds)

4. **View Results**
   - See predictions from both models side-by-side
   - View probability breakdowns for each model
   - Check the agreement indicator
   - Upload another image or reset

## 🧪 Testing with Sample Data

You can test the demo with chest X-ray images from:

- [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- [NIH Chest X-rays](https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community)
- Or use sample images from your own dataset

## 🤖 Models

### Centralized Model
- **Source**: [HuggingFace - ryefoxlime/PneumoniaDetection](https://huggingface.co/ryefoxlime/PneumoniaDetection)
- **Type**: Traditional centralized training
- **Architecture**: CNN-based classifier

### Federated Learning Model
- **Note**: For demonstration purposes, the same model is used
- **In Production**: Replace with your FL-trained model from the main platform

## 🔧 Configuration

### Changing the Federated Model

To use your own federated learning model, modify `app.py`:

```python
# Replace this line in load_models() function:
federated_model = from_pretrained_keras("ryefoxlime/PneumoniaDetection")

# With your own model loading:
federated_model = tf.keras.models.load_model('path/to/your/fl_model')
```

### Adjusting Server Settings

In `app.py`, modify the last line:

```python
app.run(
    debug=True,          # Set to False in production
    host='0.0.0.0',     # Change to '127.0.0.1' for localhost only
    port=5000           # Change port if 5000 is busy
)
```

## 🎨 Customization

### Colors and Themes

Edit `static/css/style.css` and modify the CSS variables:

```css
:root {
    --primary-color: #2563eb;      /* Main blue color */
    --secondary-color: #7c3aed;    /* Purple accent */
    --success-color: #10b981;      /* Green for normal */
    --danger-color: #ef4444;       /* Red for pneumonia */
    /* ... more variables */
}
```

### UI Text and Labels

Edit `templates/index.html` to change any text, headings, or descriptions.

## 📊 API Endpoints

### `GET /`
Returns the main HTML page

### `POST /predict`
Accepts image upload and returns predictions

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: `file` (image file)

**Response:**
```json
{
  "success": true,
  "image": "base64_encoded_image",
  "centralized": {
    "class": "Pneumonia",
    "confidence": 87.5,
    "pneumonia_probability": 87.5,
    "normal_probability": 12.5
  },
  "federated": {
    "class": "Pneumonia",
    "confidence": 85.2,
    "pneumonia_probability": 85.2,
    "normal_probability": 14.8
  }
}
```

### `GET /health`
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true
}
```

## ⚠️ Important Notes

1. **Not for Medical Use**: This is a demonstration only, not intended for actual medical diagnosis
2. **Model Performance**: The models shown are for demonstration purposes
3. **Resource Usage**: Loading models requires significant memory
4. **First Request**: The first prediction might be slower as models initialize

## 🐛 Troubleshooting

### Port Already in Use
```bash
# Change port in app.py or kill the process using port 5000
lsof -ti:5000 | xargs kill -9
```

### Models Not Loading
- Ensure you have a stable internet connection (models download from HuggingFace)
- Check you have enough RAM (at least 4GB free)
- Try running with `--no-cache-dir` flag during pip install

### Import Errors
```bash
# Make sure you're in the virtual environment
# Reinstall requirements
pip install -r requirements.txt --force-reinstall
```

### TensorFlow/Keras Issues
If you encounter Keras version issues:
```bash
pip install tf_keras==2.15.0
```

## 🚀 Next Steps

1. **Integrate Real FL Model**: Replace the demo federated model with your actual FL-trained model
2. **Add Metrics**: Include performance metrics, training history, etc.
3. **Dockerize**: Create a Docker container for easy deployment
4. **Add Authentication**: Implement user authentication if needed
5. **Deploy**: Deploy to cloud platforms (AWS, GCP, Azure, Heroku)

## 📝 License

This demo is part of the Federated Learning platform project.

## 🤝 Contributing

Feel free to enhance this demo with:
- Additional model comparison features
- Better visualizations
- More detailed explanations
- Performance metrics display

## 📧 Support

For issues or questions about this demo, please refer to the main project documentation.

---

**Built with ❤️ for demonstrating Federated Learning in Healthcare**
