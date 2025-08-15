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
    os.path.join(os.getcwd(), "test_images/0.png"),
    os.path.join(os.getcwd(), "test_images/1.png"),
    os.path.join(os.getcwd(), "test_images/2.png"),
    # Add more test images here
])
def test_recognize_digits(classifier, image_path):
    """Test recognizing digits with a single input"""
    x = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)[0]
    # Assuming the file name is in format 'digit.png'
    expected_digit = int(os.path.basename(image_path)[0])
    assert prediction == expected_digit

def test_recognize_digits_accuracy(classifier):
    """Test recognizing digits with multiple inputs and check accuracy"""
    x_test, y_test = load_mnist_test_data()
    predictions = []
    for image in x_test[:10]:  # Test with first 10 images
        image = Image.fromarray(image).convert('L').resize((28, 28))
        images = np.array(image)
        prediction = classifier(images=images)[0]
        predictions.append(prediction)

    accuracy = sum(1 for pred, actual in zip(predictions, y_test[:10]) if pred == actual) / len(y_test[:10])
    assert accuracy > 0.95

def test_recognize_digits_invalid_input(classifier):
    """Test recognizing digits with invalid input"""
    images = np.random.rand(28 * 28)
    with pytest.raises(ValueError):
        classifier(images=images)

# Add more tests here
