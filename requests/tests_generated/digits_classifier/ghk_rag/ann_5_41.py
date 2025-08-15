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
    x = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    # Assuming the correct label is stored in the filename
    expected_label = int(os.path.basename(image_path)[0])
    assert prediction == expected_label

def test_recognize_multiple_digits(classifier):
    """
    Test the classifier with multiple digits from MNIST dataset.

    Args:
        classifier (ClassifyDigits): The digit recognition model.
    """
    x_test, y_test = load_mnist_test_data()
    num_correct = 0
    for i in range(10):  # Testing first 10 images
        image = x_test[i]
        prediction = classifier(images=np.array([image]))
        if prediction == [y_test[i]]:
            num_correct += 1

    accuracy = (num_correct / 10) * 100
    assert accuracy > 95, f"Accuracy {accuracy} is less than expected"

def test_recognize_random_strings(classifier):
    """
    Test the classifier with random string data.

    Args:
        classifier (ClassifyDigits): The digit recognition model.
    """
    # Generate some random images that are not digits
    for _ in range(10):
        image = np.random.randint(low=0, high=256, size=(28, 28), dtype=np.uint8)
        prediction = classifier(images=image)

        assert isinstance(prediction[0], int) and 0 <= prediction[0] < 10

def test_model_loading():
    """
    Test if the model is loaded correctly.
    """
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
