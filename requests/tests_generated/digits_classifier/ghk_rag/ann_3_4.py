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
def classify_digits_instance():
    """Create an instance of the ClassifyDigits class."""
    return ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path_to_image_0.png",
    "path_to_image_1.png",
    # Add more image paths here...
])
def test_recognize_digit(classify_digits_instance, image_path):
    """Test if the function can recognize a single digit."""
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classify_digits_instance(images=images)
    assert isinstance(prediction[0], int)

def test_recognize_multiple_digits(classify_digits_instance):
    """Test if the function can recognize multiple digits."""
    x_test, y_test = load_mnist_test_data()
    correct_predictions = 0
    for i in range(10): # Test with at least ten different inputs
        image = x_test[i]
        images = np.array(image).reshape((1, 28, 28))
        prediction = classify_digits_instance(images=images)
        if prediction[0] == y_test[i]:
            correct_predictions += 1

    accuracy = (correct_predictions / 10) * 100
    assert accuracy > 95

def test_recognize_random_string(classify_digits_instance):
    """Test if the function can handle random string data."""
    # Create a random image with noise
    img_array = np.random.randint(0, 256, size=(28, 28), dtype=np.uint8)
    images = np.array(img_array).reshape((1, 28, 28))
    
    prediction = classify_digits_instance(images=images)

def test_model_load():
    """Test if the model is loaded correctly."""
    assert os.path.exists(model_path) == True
