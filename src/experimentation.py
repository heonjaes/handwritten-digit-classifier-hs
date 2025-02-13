import os
import tensorflow as tf
import matplotlib.pyplot as plt
import mlflow
import mlflow.keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from extraction import load_mnist_data_raw
from transformation import process_data, augment_data

def build_simple_dnn():
    model = tf.keras.models.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.1),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def build_deep_dnn():
    model = tf.keras.models.Sequential([
        layers.InputLayer(input_shape=(28, 28, 1)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def save_plot(fig, filename):
    img_dir = '../data/img'
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)  # Create the directory if it doesn't exist
    fig.savefig(os.path.join(img_dir, filename))
    print(f"Plot saved to {os.path.join(img_dir, filename)}")

def main():
    """
    Main function to load, preprocess, augment, and train the MNIST dataset.
    """
    # Load and preprocess data
    (x_train_raw, y_train), (x_test_raw, y_test) = load_mnist_data_raw()
    x_train, x_test = process_data(x_train_raw, x_test_raw)

    # Augment the training data using the data generator
    datagen_1 = augment_data(x_train, rotation_range=6)
    datagen_2 = augment_data(x_train, rotation_range=12)  # Different rotation range

    # Display basic information about the dataset
    print(f"Training data shape (processed): {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape (processed): {x_test.shape}, Test labels shape: {y_test.shape}")

    # MLflow experiment logging
    mlflow.set_experiment("mnist_model_training")
    mlflow.start_run()

    # Simple DNN model with rotation range 6
    with mlflow.start_run(nested=True, run_name="simple_dnn_rotation_6"):
        model_1 = build_simple_dnn()
        print("\nModel 1 Summary:")
        model_1.summary()

        history_1 = model_1.fit(datagen_1.flow(x_train, y_train, batch_size=32),
                                epochs=10,
                                validation_data=(x_test, y_test))

        test_loss, test_acc = model_1.evaluate(x_test, y_test)
        print(f"Model 1 Test accuracy: {test_acc:.4f}")

        # Log results to MLflow
        mlflow.log_param("model_type", "simple_dnn")
        mlflow.log_param("rotation_range", 6)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.keras.log_model(model_1, "simple_dnn_model_6")

    # Simple DNN model with rotation range 12
    with mlflow.start_run(nested=True, run_name="simple_dnn_rotation_12"):
        model_2 = build_simple_dnn()
        print("\nModel 2 Summary:")
        model_2.summary()

        history_2 = model_2.fit(datagen_2.flow(x_train, y_train, batch_size=32),
                                epochs=10,
                                validation_data=(x_test, y_test))

        test_loss, test_acc = model_2.evaluate(x_test, y_test)
        print(f"Model 2 Test accuracy: {test_acc:.4f}")

        # Log results to MLflow
        mlflow.log_param("model_type", "simple_dnn")
        mlflow.log_param("rotation_range", 12)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.keras.log_model(model_2, "simple_dnn_model_12")

    # Deep DNN model with rotation range 6
    with mlflow.start_run(nested=True, run_name="deep_dnn_rotation_6"):
        model_3 = build_deep_dnn()
        print("\nModel 3 Summary:")
        model_3.summary()

        history_3 = model_3.fit(datagen_1.flow(x_train, y_train, batch_size=32),
                                epochs=10,
                                validation_data=(x_test, y_test))

        test_loss, test_acc = model_3.evaluate(x_test, y_test)
        print(f"Model 3 Test accuracy: {test_acc:.4f}")

        # Log results to MLflow
        mlflow.log_param("model_type", "deep_dnn")
        mlflow.log_param("rotation_range", 6)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.keras.log_model(model_3, "deep_dnn_model_6")

    # Deep DNN model with rotation range 12
    with mlflow.start_run(nested=True, run_name="deep_dnn_rotation_12"):
        model_4 = build_deep_dnn()
        print("\nModel 4 Summary:")
        model_4.summary()

        history_4 = model_4.fit(datagen_2.flow(x_train, y_train, batch_size=32),
                                epochs=10,
                                validation_data=(x_test, y_test))

        test_loss, test_acc = model_4.evaluate(x_test, y_test)
        print(f"Model 4 Test accuracy: {test_acc:.4f}")

        # Log results to MLflow
        mlflow.log_param("model_type", "deep_dnn")
        mlflow.log_param("rotation_range", 12)
        mlflow.log_metric("accuracy", test_acc)
        mlflow.keras.log_model(model_4, "deep_dnn_model_12")

    # End MLflow logging
    mlflow.end_run()

    # Plot training and validation accuracy over epochs for each model
    fig, ax = plt.subplots()
    ax.plot(history_1.history['accuracy'], label='Training Accuracy (6 deg)')
    ax.plot(history_1.history['val_accuracy'], label='Validation Accuracy (6 deg)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy - Model 1')
    ax.legend()
    # save_plot(fig, 'accuracy_plot_model_1.png')

if __name__ == "__main__":
    main()
