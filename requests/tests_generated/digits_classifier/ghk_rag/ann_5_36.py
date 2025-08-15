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

    # Get the predicted digit from the classifier
    prediction = classifier(images=images)[0]

    # Assuming we have a way to get the actual label for this image (e.g., filename)
    expected_label = int(os.path.basename(image_path).split('.')[0])

    assert prediction == expected_label

def test_recognize_digits_bulk(classifier):
    """
    Test the classifier with multiple images from MNIST dataset.
    
    :param classifier: A ClassifyDigits instance
    """
    # Load and preprocess the image data in bulk using MNIST dataset
    x_test, y_test = load_mnist_test_data()
    test_images = np.array([img.reshape(28 * 28) for img in x_test[:10]]) / 255.0

    # Get predicted digits from the classifier
    predictions = [int(np.argmax(prediction)) for prediction in model.predict(test_images)]

    assert len(predictions) == 10, "Expected exactly 10 images to be classified"

def test_accuracy(classifier):
    """
    Test if the classifier recognizes over 95% of input as digits correctly.
    
    :param classifier: A ClassifyDigits instance
    """
    # Load and preprocess a larger set of image data from MNIST dataset for accuracy testing
    x_test, y_test = load_mnist_test_data()
    test_images = np.array([img.reshape(28 * 28) for img in x_test[:100]]) / 255.0

    # Get predicted digits from the classifier
    predictions = [int(np.argmax(prediction)) for prediction in model.predict(test_images)]

    accuracy = sum(predictions == y_test[:100]) / len(y_test[:100])

    assert accuracy > 0.95, f"Expected an accuracy of over 95%, but got {accuracy:.2f}"
