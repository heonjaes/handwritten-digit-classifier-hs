from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import os

# Initialize FastAPI app
app = FastAPI()

# Serve static files (HTML, JS, CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load the trained model at startup
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "models", "dnn_mnist_model.h5"))
model = tf.keras.models.load_model(MODEL_PATH)

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend."""
    with open("static/index.html", "r") as f:
        return f.read()

def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image for model prediction:
    - Normalize pixel values to [0, 1]
    - Reshape to (28, 28, 1) for CNN input
    """
    img = img / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

def image_to_base64(img: Image) -> str:
    """Convert a PIL Image to base64-encoded string."""
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.post("/predict")
async def predict(image: str = Form(...)):
    """Receive image, process it, and return predictions."""
    try:
        # Decode the base64 image string
        img_data = base64.b64decode(image)
        raw_img = Image.open(BytesIO(img_data))  # Open the raw image

        # Convert to grayscale and clean up background
        gray_img = raw_img.convert("L")  # Grayscale
        gray_img = gray_img.point(lambda p: p > 200 and 255)  # Set background to white

        # Resize to 28x28 (MNIST expected size)
        resized_img = gray_img.resize((28, 28))

        # Convert to numpy array and preprocess
        img_array = np.array(resized_img)
        img_array = preprocess_image(img_array)  # Normalize and reshape

        # Make the prediction
        prediction = model.predict(img_array)[0]  # Get probabilities for each digit (0-9)
        predicted_label = int(np.argmax(prediction))  # Most likely digit
        probabilities = prediction.tolist()  # Convert to list for JSON response

        # Convert images to base64
        grayscale_image_b64 = image_to_base64(gray_img)
        processed_image_b64 = image_to_base64(
            Image.fromarray((img_array[0, :, :, 0] * 255).astype(np.uint8))
        )

        # Return result as JSON with base64-encoded images
        return {
            "label": predicted_label,
            "probabilities": probabilities,
            "grayscale_image_base64": grayscale_image_b64,
            "processed_image_base64": processed_image_b64
        }

    except Exception as e:
        print(f"Error during prediction: {e}")  # Log the error for debugging
        raise HTTPException(status_code=400, detail="Invalid image data or processing error")
