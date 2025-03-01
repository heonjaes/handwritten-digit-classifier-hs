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
        os.makedirs(img_dir)
    fig.savefig(os.path.join(img_dir, filename))
    mlflow.log_artifact(os.path.join(img_dir, filename))

def plot_history(history, filename):
    fig, ax = plt.subplots()
    ax.plot(history.history['accuracy'], label='Training Accuracy')
    ax.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Training and Validation Accuracy')
    ax.legend()
    save_plot(fig, filename)

def main():
    (x_train_raw, y_train), (x_test_raw, y_test) = load_mnist_data_raw()
    x_train, x_test = process_data(x_train_raw, x_test_raw)
    datagen_1 = augment_data(x_train, rotation_range=6)
    datagen_2 = augment_data(x_train, rotation_range=12)

    mlflow.set_experiment("mnist_model_training")
    mlflow.start_run()

    for model_func, model_name in [(build_simple_dnn, "simple_dnn"), (build_deep_dnn, "deep_dnn")]:
        for rotation_range, datagen in [(6, datagen_1), (12, datagen_2)]:
            with mlflow.start_run(nested=True, run_name=f"{model_name}_rotation_{rotation_range}"):
                model = model_func()
                history = model.fit(datagen.flow(x_train, y_train, batch_size=32),
                                    epochs=10,
                                    validation_data=(x_test, y_test))
                test_loss, test_acc = model.evaluate(x_test, y_test)
                mlflow.log_param("model_type", model_name)
                mlflow.log_param("rotation_range", rotation_range)
                mlflow.log_metric("accuracy", test_acc)
                mlflow.keras.log_model(model, f"{model_name}_model_{rotation_range}")
                plot_history(history, f"accuracy_plot_{model_name}_{rotation_range}.png")
    mlflow.end_run()

if __name__ == "__main__":
    main()
