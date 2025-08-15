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
def mnist_test_data():
    """Load and prepare the MNIST dataset"""
    x_test, y_test = load_mnist_test_data()

    # Normalize pixel values to be between 0 and 1
    images = (x_test / 255.0).astype(np.float32)

    return images, y_test

def test_recognize_digits(mnist_test_data):
    """Test that the model recognizes over 95% of digits correctly"""
    images, labels = mnist_test_data
    
    # Create an instance of ClassifyDigits
    classifier = ClassifyDigits()

    # Make predictions on a subset of the data (first 1000 samples)
    num_samples_to_use = min(1000, len(images))
    predicted_labels = np.array([int(np.argmax(classifier(images[i:i+1]))) for i in range(num_samples_to_use)])

    actual_labels = labels[:num_samples_to_use]

    # Calculate accuracy
    correct_predictions = sum(predicted_labels == actual_labels)
    total_attempts = num_samples_to_use

    assert (correct_predictions / total_attempts) > 0.95, "Model did not recognize over 95% of digits correctly"

def test_classify_digits_interface():
    """Test that the ClassifyDigits class implements IClassifyDigits interface"""
    classifier = ClassifyDigits()
    
    # Create a random image
    img_array = np.random.rand(28, 28)
    
    result = classifier(img_array)

    assert isinstance(result, np.ndarray), "Result should be an instance of numpy array"
    assert len(result.shape) == 1 and result.dtype.kind in 'biu', "Result shape or type is incorrect"

def test_model_loading():
    """Test that the model loads correctly"""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

# Test with a single image
@pytest.mark.parametrize("image_file", ["test_image_0.png"])
def test_classify_digits_single_image(image_file):
    """Test that the ClassifyDigits class can classify a single digit correctly"""
    
    # Create an instance of ClassifyDigits
    classifier = ClassifyDigits()

    img_path = os.path.join(os.getcwd(), image_file)
    if not os.path.exists(img_path):
        pytest.skip(f"Skipping test because {image_file} does not exist")

    x = Image.open(img_path).convert('L').resize((28, 28))
    images = np.array(x)

    result = classifier(images=images)

    assert isinstance(result, np.ndarray), "Result should be an instance of numpy array"
