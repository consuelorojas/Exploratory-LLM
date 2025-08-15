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
    """Fixture to create a ClassifyDigits instance."""
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image.

    Args:
        classifier (ClassifyDigits): The digit recognition model.
        image_path (str): Path to a single-digit number image file.
    """
    x = np.array(Image.open(image_path).convert('L').resize((28, 28)))
    prediction = classifier(images=x)
    assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9

def test_recognize_digits_accuracy(classifier):
    """
    Test the accuracy of the digit recognition model.

    Args:
        classifier (ClassifyDigits): The digit recognition model.
    """
    x_test, y_test = load_mnist_test_data()
    predictions = []
    
    # Preprocess images
    for image in x_test[:10]:
        img_array = np.array(image).reshape(1, 28 * 28)
        prediction = classifier(images=img_array / 255.0)  # Normalize the input data
        predictions.append(prediction[0])
        
    accuracy = sum([p == y for p, y in zip(predictions, y_test[:10])]) / len(y_test[:10])
    
    assert accuracy > 0.95

def test_invalid_input(classifier):
    """
    Test that an invalid input raises a ValueError.

    Args:
        classifier (ClassifyDigits): The digit recognition model.
    """
    with pytest.raises(ValueError):
        # Try passing in something other than a numpy array
        classifier(images="not_a_numpy_array")

# Example of using the asterisk (*) for multiple steps
def test_multiple_inputs(classifier):
    """Test that the function can handle multiple inputs."""
    *images = [
        np.array(Image.open("path/to/image1.png").convert('L').resize((28, 28))),
        np.array(Image.open("path/to/image2.png").convert('L').resize((28, 28)))
    ]
    
    predictions = []
    for image in images:
        prediction = classifier(images=image)
        assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9
