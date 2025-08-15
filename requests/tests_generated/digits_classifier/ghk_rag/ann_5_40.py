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
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digit(classify_digits_instance, tmpdir):
    """Test recognizing a single digit from an image file."""
    x_test, y_test = load_mnist_test_data()
    
    for i in range(10):  # Test with at least ten different inputs
        img_array = x_test[i]
        
        # Save the array as an image to test the function's ability to read images
        img_path = os.path.join(tmpdir, f"image_{i}.png")
        Image.fromarray(img_array).save(img_path)
        
        prediction = classify_digits_instance(np.array(Image.open(img_path)))
        assert int(prediction) == y_test[i]

def test_recognize_multiple_digits(classify_digits_instance):
    """Test recognizing multiple digits from an array."""
    x_test, y_test = load_mnist_test_data()
    
    # Select at least ten different inputs
    images_to_test = np.array([x_test[i] for i in range(10)])
    
    predictions = classify_digits_instance(images_to_test)
    assert len(predictions) == 10
    
    correct_count = sum(int(prediction) == y_test[i] for i, prediction in enumerate(predictions))
    accuracy = (correct_count / len(predictions)) * 100
    assert accuracy > 95

def test_load_model():
    """Test loading the model from a file."""
    try:
        tf.keras.models.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

# Example of using parametrize with multiple inputs and expected outputs
@pytest.mark.parametrize("input_array,expected_output", [
    (np.array([[1]]), np.array([0])),  # Replace these values with actual test data
    (np.array([[2]]), np.array([1])),
])
def test_classify_digits(classify_digits_instance, input_array, expected_output):
    """Test the ClassifyDigits function."""
    output = classify_digits_instance(input_array)
    assert np.all(output == expected_output)

# Example of using fixture to create a dataset
@pytest.fixture
def mnist_test_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return x_test[:10], y_test[:10]

def test_classify_digits_with_mnist(classify_digits_instance, mnist_test_data):
    """Test the ClassifyDigits function with MNIST data."""
    images_to_test, expected_outputs = mnist_test_data
    predictions = classify_digits_instance(images_to_test)
    
    correct_count = sum(int(prediction) == output for prediction, output in zip(predictions, expected_outputs))
    accuracy = (correct_count / len(expected_outputs)) * 100
    
    assert accuracy > 95

# Example of using pytest.mark.parametrize to test with different inputs
@pytest.mark.parametrize("image_array", [
    np.array([[1]]),  # Replace these values with actual test data
    np.array([[2]]),
])
def test_classify_digits_with_parametrize(classify_digits_instance, image_array):
    """Test the ClassifyDigits function."""
    output = classify_digits_instance(image_array)
    assert isinstance(output, np.ndarray)

# Example of using pytest.mark.parametrize to test with different inputs and expected outputs
@pytest.mark.parametrize("image_array,expected_output", [
    (np.array([[1]]), 0),
    (np.array([[2]]), 1),
])
def test_classify_digits_with_parametrize_and_expected(classify_digits_instance, image_array, expected_output):
    """Test the ClassifyDigits function."""
    output = classify_digits_instance(image_array)
    assert int(output) == expected_output
