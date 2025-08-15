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
    return ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image.

    :param classifier: A ClassifyDigits instance.
    :param image_path: Path to a single-digit number image (0-9).
    """
    x = np.array(Image.open(image_path).convert('L').resize((28, 28)))
    prediction = classifier(images=x)
    # Assuming the correct label is stored in an external file or database
    with open(os.path.splitext(image_path)[0] + ".label", "r") as f:
        expected_label = int(f.read())
    assert prediction == [expected_label]

def test_recognize_digits_accuracy(classifier):
    """
    Test the classifier's accuracy on a dataset.

    :param classifier: A ClassifyDigits instance.
    """
    x_test, y_test = load_mnist_test_data()
    predictions = []
    for image in x_test[:10]:  # Use first 10 images from test data
        image_array = np.array(image).reshape(1, 28 * 28)
        prediction = classifier(images=image_array / 255.0)  # Normalize input
        predictions.append(prediction[0])
    
    accuracy = sum([p == y for p, y in zip(predictions, y_test[:10])]) / len(y_test[:10])
    assert accuracy > 0.95

def test_load_model():
    """
    Test that the model is loaded correctly.
    """
    # Check if the model exists at the specified path
    assert os.path.exists(model_path)
