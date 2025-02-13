import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from extraction import load_mnist_data_raw
from sklearn.preprocessing import StandardScaler

def process_data(x_train_raw, x_test_raw):
    """
    Process the MNIST data by normalizing, reshaping, and applying augmentations.
    """
    # Normalize pixel values to [0, 1]
    x_train = x_train_raw / 255.0
    x_test = x_test_raw / 255.0

    # Reshape to (28, 28, 1) for CNN input
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    return x_train, x_test

def augment_data(x_train, rotation_range=12):
    """
    Augment the training data using transformations like rotations, shifts, and zooms.
    """
    datagen = ImageDataGenerator(
        rotation_range=12,      # Random rotation between -12 and 12 degrees
        width_shift_range=0.1,  # Randomly translate images horizontally by 10%
        height_shift_range=0.1, # Randomly translate images vertically by 10%
        fill_mode='nearest'     # How to fill the pixels after transformation
    )
    
    # Fit the data generator to the training data
    datagen.fit(x_train)
    return datagen


def main():
    """
    Main function to load, preprocess, and check the MNIST dataset.
    """
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_mnist_data_raw()
    
    # Augment the training data
    datagen = augment_data(x_train)
    
    # Display basic information about the dataset
    print(f"Training data shape: {x_train.shape}, Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Test labels shape: {y_test.shape}")
    print(f"Sample training label (one-hot encoded): {y_train[0]}")
    print(f"Sample training image (flattened):\n{x_train[0]}")
    

if __name__ == "__main__":
    main()
