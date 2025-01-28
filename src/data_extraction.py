import tensorflow as tf

def load_mnist_data():
    """
    Load and preprocess the MNIST dataset.
    Normalizes pixel values to the range [0, 1].

    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize data
    return (x_train, y_train), (x_test, y_test)

def main():
    """
    Main function to load and check the MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Display basic information about the dataset
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
    print(f"Sample training label: {y_train[0]}")
    print(f"Sample training image (normalized pixel values):\n{x_train[0]}")

if __name__ == "__main__":
    main()
