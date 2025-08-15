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
    # Add more image paths as needed
])
def test_recognize_digits(classifier, image_path):
    """
    Test that the classifier recognizes digits correctly.

    :param classifier: A ClassifyDigits instance.
    :param image_path: Path to an image file containing a digit.
    """
    x = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)

    # Assuming the correct label is stored in the filename
    expected_label = int(os.path.basename(image_path)[0])
    assert prediction == expected_label

def test_recognize_digits_accuracy(classifier):
    """
    Test that the classifier recognizes over 95% of digits correctly.

    :param classifier: A ClassifyDigits instance.
    """
    x_test, y_test = load_mnist_test_data()
    correct_count = 0
    total_count = len(x_test)

    for i in range(total_count):
        image = x_test[i]
        images = np.array(image) / 255.0                 # normalize
        images = images.reshape(-1, 28 * 28)             # flatten

        prediction = classifier(images=images)
        if int(prediction[0]) == y_test[i]:
            correct_count += 1

    accuracy = (correct_count / total_count) * 100
    assert accuracy > 95.0

def test_classify_digits_interface(classifier):
    """
    Test that the ClassifyDigits instance conforms to the IClassifyDigits interface.

    :param classifier: A ClassifyDigits instance.
    """
    images = np.random.rand(1, 28 * 28)
    prediction = classifier(images=images)

    assert isinstance(prediction, np.ndarray)
    assert len(prediction.shape) == 1
