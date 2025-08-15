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
])
def test_load_image(image_path):
    """Test loading an image from file path."""
    img = Image.open(image_path).convert('L').resize((28, 28))
    assert isinstance(img, PIL.Image.Image)

@pytest.mark.parametrize("image_array", [
    np.random.randint(0, 256, size=(1, 28, 28), dtype=np.uint8),
])
def test_classify_digits(classifier: ClassifyDigits, image_array):
    """Test classifying a single digit."""
    prediction = classifier(image_array)
    assert isinstance(prediction, np.ndarray)

@pytest.mark.parametrize("image_arrays", [
    [np.random.randint(0, 256, size=(1, 28, 28), dtype=np.uint8) for _ in range(10)],
])
def test_classify_multiple_digits(classifier: ClassifyDigits, image_arrays):
    """Test classifying multiple digits."""
    predictions = []
    for img_array in image_arrays:
        prediction = classifier(img_array)
        assert isinstance(prediction, np.ndarray)
        predictions.append(prediction)

@pytest.mark.slow
def test_recognition_accuracy(classifier: ClassifyDigits):
    """
    Test the recognition accuracy of the model.

    This test loads MNIST test data and checks if at least 95% of digits are recognized correctly.
    """
    x_test, y_test = load_mnist_test_data()
    
    # Preprocess images
    images = np.array([img.reshape(28 * 28) for img in x_test])
    labels = y_test
    
    predictions = classifier(images)
    
    accuracy = sum(predictions == labels) / len(labels)
    assert accuracy > 0.95

def test_model_loading():
    """Test loading the model from file."""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

# Example of using Gherkin syntax in a Python docstring
class TestDigitRecognition:
    def test_recognition(self):
        """
        Given a function capable of recognizing digit strings
        And a dataset containing various single-digit numbers (0 through 9) and other random string data
        When I test this function with at least ten different inputs from my dataset
        Then it should recognize over 95% of the input as digits correctly.
        
        This is implemented in `test_recognition_accuracy`.
        """
