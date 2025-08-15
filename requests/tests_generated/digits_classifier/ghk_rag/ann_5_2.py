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
    """Fixture to create a ClassifyDigits instance"""
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image
    :param classifier: ClassifyDigits instance from fixture
    :param image_path: Path to a single-digit number image file (0-9)
    """
    # Load and preprocess the image data
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)

    # Get prediction for this input
    predicted_digit = classifier(images=images)[0]

    # Assuming we have a way to get the actual digit from the file name or metadata...
    expected_digit = int(os.path.basename(image_path).split('.')[0])

    assert predicted_digit == expected_digit

def test_recognize_digits_bulk(classifier):
    """
    Test the classifier with multiple images and verify accuracy
    :param classifier: ClassifyDigits instance from fixture
    """
    # Load MNIST test data for testing purposes
    x_test, y_test = load_mnist_test_data()

    correct_count = 0

    # Iterate over at least ten different inputs (we'll use the entire dataset)
    for i in range(len(x_test)):
        img = np.array([x_test[i]])
        
        predicted_digit = classifier(images=img)[0]
        expected_digit = y_test[i]

        if predicted_digit == expected_digit:
            correct_count += 1

    accuracy = correct_count / len(y_test)

    assert accuracy > 0.95
