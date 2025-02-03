from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import base64
from io import BytesIO

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
    # Decode the base64 image string
    img_data = base64.b64decode(image)
    img = Image.open(BytesIO(img_data))  # Open the raw image

    # Save the raw image to 'data/input' folder
    input_folder = "data/input"
    os.makedirs(input_folder, exist_ok=True)  # Create the directory if it doesn't exist
    raw_img_path = os.path.join(input_folder, "raw_image.png")
    img.save(raw_img_path)

    # Convert the image to grayscale (MNIST style: black background, white text)
    grayscale_img = img.convert("L")  # 'L' mode is for grayscale
    grayscale_img_path = os.path.join(input_folder, "grayscale_image.png")
    grayscale_img.save(grayscale_img_path)

    # Resize the image to 28x28 (MNIST expected size)
    resized_img = grayscale_img.resize((28, 28))
    processed_img_path = os.path.join(input_folder, "processed_image.png")
    resized_img.save(processed_img_path)

    # Convert image to numpy array and preprocess it (normalize and reshape)
    img_array = np.array(resized_img)
    img_array = preprocess_image(img_array)  # Same preprocessing as before

    # Make prediction
    prediction = model.predict(img_array)[0]  # Get probabilities for each digit (0-9)
    predicted_label = int(np.argmax(prediction))  # Most likely digit
    probabilities = prediction.tolist()  # Convert to list for JSON response

    return {"label": predicted_label, "probabilities": probabilities}
