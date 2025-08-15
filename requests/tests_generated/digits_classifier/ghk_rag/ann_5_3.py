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
    Test that the classifier recognizes digits correctly
    """
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    # Assuming you have a way to get the actual digit from the image path
    expected_digit = int(os.path.basename(image_path)[0])
    assert prediction == expected_digit

def test_recognize_digits_accuracy(classifier):
    """
    Test that the classifier recognizes over 95% of digits correctly
    """
    x_test, y_test = load_mnist_test_data()
    correct_count = 0
    for i in range(10): # Use at least ten different inputs from your dataset
        image = x_test[i]
        images = np.array(image) / 255.0                 # normalize
        images = images.reshape(-1, 28 * 28)             # flatten
        prediction = classifier(images=images)
        if prediction == y_test[i]:
            correct_count += 1

    accuracy = (correct_count / len(y_test[:10])) * 100
    assert accuracy > 95

def test_classify_digits_interface(classifier):
    """
    Test that the ClassifyDigits class implements the IClassifyDigits interface correctly
    """
    images = np.random.rand(1, 28*28)
    result = classifier(images=images)
    assert isinstance(result, np.ndarray)

