
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
    correct_predictions = sum(1 for pred, actual in zip(predictions, y_test[:10]) if pred == actual)
    accuracy = (correct_predictions / len(y_test[:10])) * 100
    assert accuracy > 95

def test_recognize_digits_with_random_noise(classify_digits_instance):
    """Test recognizing digits with random noise."""
    x_test, _ = load_mnist_test_data()
    noisy_images = np.array([x + np.random.randint(0, 255) for x in x_test[:10]])
    predictions = classify_digits_instance(images=noisy_images)
    assert len(predictions) == 10

def test_recognize_non_digit_input(classify_digits_instance):
    """Test recognizing non-digit input."""
    # Create a random image that is not a digit
    img_array = np.random.randint(0, 255, size=(28, 28))
    images = np.array(img_array)
    prediction = classify_digits_instance(images=images)
    assert isinstance(prediction, int)

def test_model_loading():
    """Test if the model loads correctly."""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
