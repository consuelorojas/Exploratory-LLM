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
    """Create a ClassifyDigits instance"""
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image
    """
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    # Assuming you have a way to get the actual digit from the image path
    expected_digit = int(os.path.basename(image_path)[0])
    assert prediction == [expected_digit]

def test_recognize_digits_accuracy(classifier):
    """
    Test the accuracy of the classifier with MNIST data
    """
    x_test, y_test = load_mnist_test_data()
    correct_count = 0

    for i in range(10): # Testing on first 10 images from each class (total: 100)
        image = x_test[i]
        prediction = classifier(images=np.array([image]))
        if prediction[0] == y_test[i]:
            correct_count += 1
    accuracy = correct_count / len(y_test[:100])
    assert accuracy > 0.95

def test_recognize_digits_multiple_inputs(classifier):
    """
    Test the classifier with multiple inputs at once
    """
    x_test, _ = load_mnist_test_data()
    images = np.array([x_test[i] for i in range(10)])
    predictions = classifier(images=images)
    assert len(predictions) == 10

def test_recognize_digits_invalid_input(classifier):
    """
    Test the classifier with invalid input
    """
    with pytest.raises(ValueError):
        classifier(images=np.random.rand(1, 28)) # Invalid shape or type of image data
