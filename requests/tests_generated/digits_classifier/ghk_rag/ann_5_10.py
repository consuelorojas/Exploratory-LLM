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
    """Create a ClassifyDigits instance."""
    return ClassifyDigits()

# Load MNIST test data for testing purposes
@pytest.fixture
def mnist_test_data():
    """Load and prepare the MNIST dataset for testing."""
    (x_train, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values to be between 0 and 1
    x_test_normalized = x_test / 255.0
    
    return x_test_normalized, y_test

def test_recognize_digits_correctly(classifier: ClassifyDigits, mnist_test_data):
    """Test that the classifier recognizes over 95% of digits correctly."""
    
    # Get MNIST test data
    images, labels = mnist_test_data
    
    # Select a subset of at least ten different inputs from the dataset for testing
    num_inputs_to_test = min(10, len(images))
    indices_to_test = np.random.choice(len(images), size=num_inputs_to_test, replace=False)
    
    selected_images = images[indices_to_test]
    expected_labels = labels[indices_to_test]

    # Reshape and normalize the input data for classification
    reshaped_selected_images = selected_images.reshape(-1, 28 * 28)

    predicted_digits = classifier(reshaped_selected_images)
    
    correct_predictions = np.sum(predicted_digits == expected_labels)
    accuracy_percentage = (correct_predictions / num_inputs_to_test) * 100
    
    assert accuracy_percentage > 95

def test_invalid_input(classifier: ClassifyDigits):
    """Test that the classifier raises an error when given invalid input."""
    
    # Create a deliberately incorrect image with wrong dimensions
    invalid_image = np.random.rand(10, 20)
    
    with pytest.raises(ValueError) as exc_info:
        classifier(invalid_image)

def test_empty_input(classifier: ClassifyDigits):
    """Test that the classifier raises an error when given empty input."""
    
    # Create a deliberately incorrect image (empty array)
    invalid_image = np.array([])
    
    with pytest.raises(ValueError) as exc_info:
        classifier(invalid_image)
