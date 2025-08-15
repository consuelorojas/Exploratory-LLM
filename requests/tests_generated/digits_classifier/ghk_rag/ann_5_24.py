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
    return ClassifyDigits()

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

def test_recognize_digits_bulk(classifier):
    """Test recognizing digits with multiple inputs"""
    x_test, y_test = load_mnist_test_data()
    # Select a subset of the data for testing
    indices_to_use = np.random.choice(len(x_test), size=10)
    images = [x_test[i] / 255.0 for i in indices_to_use]
    expected_digits = [y_test[i] for i in indices_to_use]

    predictions = []
    for image in images:
        # Reshape the image to match what ClassifyDigits expects
        reshaped_image = np.reshape(image, (1, -1))
        prediction = classifier(images=reshaped_image)[0]
        predictions.append(prediction)

    accuracy = sum([p == e for p, e in zip(predictions, expected_digits)]) / len(expected_digits)
    assert accuracy > 0.95

def test_model_load():
    """Test that the model is loaded correctly"""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
