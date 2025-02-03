from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import io
import keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="web"), name="static")

# Load the trained model at startup
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "dnn_mnist_model.h5"))
model = tf.keras.models.load_model(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    with open("web/index.html", "r") as f:
        return f.read()

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Process input image to match the training preprocessing:
    - Normalize pixel values to [0, 1]
    - Reshape to (28, 28, 1) for CNN input
    """
    img = img / 255.0  # Normalize
    img = img.reshape(1, 28, 28, 1)  # Reshape for model
    return img


@app.post("/predict")
async def predict(image: str = Form(...)):
    """Receive image, process, and return predictions."""
    try:
        # Decode the base64 image string
        img_data = base64.b64decode(image)
        img = Image.open(BytesIO(img_data))  # Open the raw image

        # Convert to grayscale (ensure background is white)
        img = img.convert("L")  # Grayscale (mode 'L' for luminance)
        img = img.point(lambda p: p > 200 and 255)  # Set everything but drawn content to white

        # Resize the image to 28x28 (MNIST expected size)
        resized_img = img.resize((28, 28))

        # Convert image to numpy array and preprocess it (normalize and reshape)
        img_array = np.array(resized_img)
        img_array = preprocess_image(img_array)  # Same preprocessing as before

        # Make prediction
        prediction = model.predict(img_array)[0]  # Get probabilities for each digit (0-9)
        predicted_label = int(np.argmax(prediction))  # Most likely digit
        probabilities = prediction.tolist()  # Convert to list for JSON response

        # Return result as JSON
        return {
            "label": predicted_label,
            "probabilities": probabilities,
        }
    except Exception as e:
        print("Error during prediction:", e)  # Print the error for debugging
        raise HTTPException(status_code=400, detail="Invalid image data or processing error")
