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
    # Load and preprocess a subset of MNIST data for testing purposes
    x_test, y_test = load_mnist_test_data()
    num_images_to_use = 10

    correct_count = 0
    total_count = 0

    for i in range(num_images_to_use):
        img = x_test[i]
        images = np.array(img).reshape(1, -1) / 255.0  # Normalize and flatten the image data

        prediction = classifier(images=images)[0]

        if prediction == y_test[i]:
            correct_count += 1
        total_count += 1

    accuracy = (correct_count / total_count) * 100
    assert accuracy > 95, f"Accuracy {accuracy} is less than the expected threshold of 95%"

def test_load_model():
    """
    Test that a model can be loaded from the specified path.
    
    :return:
    """
    try:
        tf.keras.models.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Failed to load model: {str(e)}")
