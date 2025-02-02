import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from extraction import load_mnist_data_raw
from transformation import process_data, augment_data
from tensorflow.keras.utils import plot_model

def build_dnn_model(input_shape):
    """
    Build a deep neural network (DNN) model.
    """
    model = tf.keras.models.Sequential([
        # Input layer specifying input shape
        layers.InputLayer(input_shape=(28, 28, 1)),

        # First Conv2D layer + MaxPooling
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Second Conv2D layer + MaxPooling
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),

        # Flatten the feature maps to 1D
        layers.Flatten(),

        # Dense layer (fully connected)
        layers.Dense(128, activation='relu'),

        # Dropout layer to prevent overfitting
        layers.Dropout(0.1),

        # Output layer with 10 neurons (one for each class)
        layers.Dense(10, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def save_plot(fig, filename):
    """
    Save the plot as an image in the specified file.
    """
    img_dir = '../data/img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)  # Create the directory if it doesn't exist
    fig.savefig(os.path.join(img_dir, filename))
    print(f"Plot saved to {os.path.join(img_dir, filename)}")

def main():
    """
    Main function to load, preprocess, augment, and train the MNIST dataset with a DNN.
    """
    # Load and preprocess data
    (x_train_raw, y_train), (x_test_raw, y_test) = load_mnist_data_raw()

    # Process the data by normalizing and reshaping
    x_train, x_test = process_data(x_train_raw, x_test_raw)

    # Augment the training data using the data generator
    datagen = augment_data(x_train)

    # Display basic information about the dataset
    print(f"Training data shape (processed): {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape (processed): {x_test.shape}, Test labels shape: {y_test.shape}")
    
    # Create the model
    model = build_dnn_model(input_shape=(28, 28, 1))

    # Display the model summary
    print("\nModel Summary:")
    model.summary()

    # Train the model with data augmentation
    history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                        epochs=10,
                        validation_data=(x_test, y_test))

    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Plot training and validation accuracy over epochs
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    save_plot(fig, 'accuracy_plot.png')

    # Plot training and validation loss over epochs
    fig, ax = plt.subplots()
    ax.plot(history.history['loss'], label='Training Loss')
    ax.plot(history.history['val_loss'], label='Validation Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    save_plot(fig, 'loss_plot.png')

    # Save the trained model to '../models' directory
    model_save_path = '../models/dnn_mnist_model.h5'
    if not os.path.exists('../models'):
        os.makedirs('../models')  # Create the directory if it doesn't exist
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()
