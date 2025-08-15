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
    "path_to_your_image_0.png",
    "path_to_your_image_1.png",
    # Add more image paths here...
])
def test_recognize_digits(classify_digits_instance, image_path):
    """Test the recognition of digits."""
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classify_digits_instance(images=images)
    assert isinstance(prediction, int)

def test_recognize_multiple_digits(classify_digits_instance):
    """Test recognizing multiple digits with over 95% accuracy."""
    x_test, y_test = load_mnist_test_data()
    
    # Select a subset of the MNIST dataset for testing
    num_samples_to_use = min(1000, len(x_test))
    indices_to_use = np.random.choice(len(x_test), size=num_samples_to_use)
    test_images = [x_test[i] / 255.0 for i in indices_to_use]
    
    # Reshape images to match the expected input shape
    reshaped_images = [image.reshape(-1, 28 * 28) for image in test_images]

    predictions = []
    actual_labels = []

    for index, (reshaped_image, label) in enumerate(zip(reshaped_images[:10], y_test[indices_to_use][:10])):
        prediction = classify_digits_instance(images=reshaped_image)
        assert isinstance(prediction, int)

        # Store the results
        predictions.append(int(np.argmax([prediction])))
        actual_labels.append(label)

    accuracy = sum([1 for pred, label in zip(predictions, actual_labels) if pred == label]) / len(actual_labels)
    
    assert accuracy > 0.95

def test_load_model():
    """Test that the model is loaded correctly."""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

