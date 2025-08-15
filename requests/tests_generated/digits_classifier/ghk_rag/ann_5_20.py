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
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths as needed for testing
])
def test_recognize_digits(classifier, image_path):
    """Test recognizing digits with a single input"""
    x = PIL.Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9

def test_recognize_digits_accuracy(classifier):
    """Test recognizing digits with a dataset"""
    x_test, y_test = load_mnist_test_data()
    
    # Select at least ten different inputs from the dataset
    num_inputs_to_test = min(10, len(x_test))
    indices_to_test = np.random.choice(len(x_test), size=num_inputs_to_test, replace=False)
    test_images = [x_test[i] for i in indices_to_test]
    expected_labels = [y_test[i] for i in indices_to_test]

    # Preprocess images
    preprocessed_images = []
    for image in test_images:
        img = PIL.Image.fromarray(image).convert('L').resize((28, 28))
        preprocessed_image = np.array(img)
        preprocessed_images.append(preprocessed_image)

    predictions = classifier(images=np.stack(preprocessed_images))

    # Calculate accuracy
    correct_predictions = sum(1 for pred, label in zip(predictions[0], expected_labels) if pred == label)
    accuracy = (correct_predictions / num_inputs_to_test) * 100

    assert accuracy > 95.0

def test_load_model():
    """Test loading the model"""
    try:
        tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        pytest.fail(f"Failed to load model: {e}")

# Example of using an asterisk (*) in place of any step keyword
@pytest.mark.parametrize("image", [
    "path/to/image1.png",
    # Add more image paths as needed for testing
])
def test_recognize_digits_with_asterisk(classifier, image):
    """Test recognizing digits with a single input"""
    x = PIL.Image.open(image).convert('L').resize((28, 28))
    images = np.array(x)
    prediction = classifier(images=images)
    assert isinstance(prediction[0], int) and 0 <= prediction[0] <= 9
