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
        
        # Save the numpy array as an image to disk temporarily
        tmp_image_path = os.path.join(tmpdir, f"image_{i}.png")
        Image.fromarray(img_array).save(tmp_image_path)
        
        images = np.array(Image.open(tmp_image_path))
        prediction = classify_digits_instance(images=images)[0]
        
        assert prediction == y_test[i]

def test_recognize_multiple_digits(classify_digits_instance):
    """Test recognizing multiple digits from the MNIST dataset."""
    x_test, y_test = load_mnist_test_data()
    
    # Select at least ten different inputs
    selected_indices = np.random.choice(len(x_test), size=10, replace=False)
    images = [x_test[i] for i in selected_indices]
    labels = [y_test[i] for i in selected_indices]

    predictions = []
    for image in images:
        img_array = np.array(image) / 255.0
        img_array = img_array.reshape(-1, 28 * 28)
        prediction = classify_digits_instance(images=img_array)[0]
        predictions.append(prediction)

    accuracy = sum(1 for pred, label in zip(predictions, labels) if pred == label) / len(labels)
    
    assert accuracy > 0.95

def test_model_loads_correctly():
    """Test that the model loads correctly from the specified path."""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")
