# Handwritten Digit Classifier Web App

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![TensorFlow](https://img.shields.io/badge/tensorflow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-4.x-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![HTML](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

This repository contains the code for a **Handwritten Digit Classifier Web App**. The app allows users to draw a digit on a canvas, and it returns a prediction based on a neural network model built using **TensorFlow**. The backend is powered by **FastAPI**, providing high performance and ease of development, while the frontend is developed using **HTML**, **CSS**, and **JavaScript**.

## Folder Structure

```
handwritten-digit-classifier/
├── app.py                # Main entry point for the FastAPI backend
├── data/                 # Contains datasets used for training the model
│   ├── img/              # Contains accuracy and loss plot images
│   ├── input/            # Contains raw, grayscaled, and scaled images from the web app canvas
├── model/                # Trained machine learning model
│   └── model.h5          # Neural network model
├── src/                  # Source code for the backend (FastAPI)
│   └── model.py          # Final training code to generate DNN
│   └── transformation.py # Trasformations applied to MNIST dataset
│   └── data_exploration.py # Data exploration and experiments run in Notebook
├── web/                  # Frontend code
│   ├── index.html        # HTML file for the web interface
│   ├── style.css         # Custom styles for the web app
│   ├── app.js            # JavaScript logic for frontend functionality
│   └── ...               # Additional static files for the frontend
└── requirements.txt      # Backend dependencies
```

## Features

- **Canvas for Drawing**: Users can draw a digit on the canvas.
- **Prediction**: After drawing, users can click the "Predict" button to get the predicted digit along with the probability distribution.
- **Clear Canvas**: A button to clear the drawing area and start a new prediction.
- **Probability Chart**: Visualizes the prediction probabilities using a bar chart.

## Tech Stack

### Backend

- **Python**: The backend is written in Python, a powerful programming language known for its simplicity and versatility.
- **TensorFlow**: The neural network model is built using TensorFlow, a popular machine learning framework. The model is trained to recognize handwritten digits (MNIST dataset).
- **FastAPI**: A modern, fast web framework for building APIs with Python. FastAPI is used to serve the model and handle prediction requests.

### Frontend

- **HTML, CSS, JavaScript**: The frontend is built with standard web technologies, including HTML, CSS, and JavaScript.
- **Bootstrap 5**: For responsive design and easy styling.
- **Chart.js**: Used for visualizing prediction probabilities as a bar chart.

## Getting Started

### Prerequisites

- Python 3.7 or above
- Node.js and npm (for frontend dependencies)
- FastAPI for the backend

### Installation

1. **Clone the repository**

```bash
git clone <repo-url>
cd handwritten-digit-classifier
```

2. **Install backend dependencies**

```bash
pip install -r requirements.txt
```

3. **Install frontend dependencies**

```bash
cd web
npm install
```

4. **Run the backend server**

```bash
uvicorn app:app --reload
```

5. **Run the frontend**

```bash
cd web
npm start
```

The app should now be accessible at `http://localhost:3000`.

## Usage

- **Draw a digit**: Use the canvas to draw a digit between 0-9.
- **Predict the digit**: Click on "Predict" to get the predicted digit and its associated probabilities.
- **Clear the canvas**: Click "Clear" to reset the drawing area and prepare for a new prediction.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

