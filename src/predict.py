import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64
import os

# Get the absolute path to the model file using os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Project root directory
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'dnn_mnist_model.h5')

# Load the model once (avoid reloading on each request)
model = tf.keras.models.load_model(MODEL_PATH)

def predict_image(image_data: str) -> dict:
    # Decode base64 image data
    img_str = image_data.split(",")[1]
    img_bytes = base64.b64decode(img_str)
    img = Image.open(io.BytesIO(img_bytes)).convert("L")

    # Preprocess the image
    img = img.resize((28, 28))
    img = np.array(img) / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input

    # Predict with the model
    prediction = model.predict(img)
    predicted_label = np.argmax(prediction)
    confidence = np.max(prediction)

    return {"label": int(predicted_label), "confidence": float(confidence)}
