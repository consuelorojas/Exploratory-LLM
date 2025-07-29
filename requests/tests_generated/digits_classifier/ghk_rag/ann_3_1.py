
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
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path_to_your_image_0.png",
    "path_to_your_image_1.png",
    # Add more image paths here...
])
def test_recognize_digits(classify_digits_instance, image_path):
    """Test the recognition of digits."""
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classify_digits_instance(images=images)
    # Assuming you have a way to get the actual digit from the image path
    expected_digit = int(os.path.basename(image_path)[0])
    assert prediction == expected_digit

def test_recognize_multiple_digits(classify_digits_instance):
    """Test recognizing multiple digits."""
    x_test, y_test = load_mnist_test_data()
    images = np.array([x / 255.0 for x in x_test[:10]])
    predictions = classify_digits_instance(images=images)
    assert len(predictions) == 10

def test_accuracy(classify_digits_instance):
    """Test the accuracy of digit recognition."""
    x_test, y_test = load_mnist_test_data()
    images = np.array([x / 255.0 for x in x_test[:100]])
    predictions = classify_digits_instance(images=images)
    correct_predictions = sum(1 for pred, actual in zip(predictions, y_test[:100]) if pred == actual)
    accuracy = correct_predictions / len(y_test[:100])
    assert accuracy > 0.95

def test_invalid_input(classify_digits_instance):
    """Test the handling of invalid input."""
    with pytest.raises(ValueError):
        classify_digits_instance(images=np.array([1, 2, 3]))  # Invalid shape
