import tensorflow as tf

import pytest
from your_module import ClassifyDigits, model_path
import numpy as np
from PIL import Image
import os

# Load MNIST test data for testing purposes
def load_mnist_test_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test, y_test

@pytest.fixture
def classifier():
    """Create a ClassifyDigits instance."""
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image.
    
    :param classifier: A ClassifyDigits instance
    :param image_path: Path to a single-digit number image file (0-9)
    """
    # Load and preprocess the image data
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)

    # Get prediction from classifier
    predicted_digit = classifier(images=images)[0]

    # Assuming we have a way to get the actual digit value (e.g., filename or label)
    expected_digit = int(os.path.basename(image_path).split('.')[0])

    assert predicted_digit == expected_digit

def test_recognize_digits_bulk(classifier):
    """
    Test the classifier with multiple images from MNIST dataset.
    
    :param classifier: A ClassifyDigits instance
    """
    # Load and preprocess MNIST data for testing purposes
    x_test, y_test = load_mnist_test_data()

    correct_count = 0

    for i in range(10):  # Test with at least ten different inputs from the dataset
        img = x_test[i]
        images = np.array(img).reshape((1, 28 * 28))

        predicted_digit = classifier(images=images)[0]

        if predicted_digit == y_test[i]:
            correct_count += 1

    accuracy = (correct_count / len(y_test[:10])) * 100
    assert accuracy > 95.0
