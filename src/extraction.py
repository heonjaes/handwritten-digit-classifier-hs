import tensorflow as tf

# Load the MNIST dataset as-is (no preprocessing yet)
def load_mnist_data_raw():
    """
    Load the MNIST dataset without any preprocessing.
    """
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return (x_train, y_train), (x_test, y_test)

def main():
    """
    Main function to load and check the MNIST dataset.
    """
    (x_train, y_train), (x_test, y_test) = load_mnist_data_raw()
    
    # Display basic information about the dataset
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
    print(f"Sample training label: {y_train[0]}")
    print(f"Sample training image (normalized pixel values):\n{x_train[0]}")

if __name__ == "__main__":
    main()
