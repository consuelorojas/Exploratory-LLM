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
    yield ClassifyDigits()

@pytest.mark.parametrize("image_path", [
    "path/to/image1.png",
    "path/to/image2.png",
    # Add more image paths here...
])
def test_recognize_digits(classifier, image_path):
    """
    Test the classifier with an individual image.
    
    :param classifier: A ClassifyDigits instance
    :param image_path: Path to a single-digit number image file (0-9)
    """
    # Load and preprocess the image data
    img = Image.open(image_path).convert('L').resize((28, 28))
    images = np.array(img)

    # Make predictions using the classifier
    prediction = classifier(images=images)[0]

    # Verify that the predicted digit is correct (assuming we know the ground truth)
    expected_digit = int(os.path.basename(image_path)[-5])  # Assuming filename format: "imageX.png"
    assert prediction == expected_digit

def test_recognize_digits_bulk(classifier):
    """
    Test the classifier with a bulk dataset.
    
    :param classifier: A ClassifyDigits instance
    """
    x_test, y_test = load_mnist_test_data()

    # Select at least 10 different inputs from our MNIST test data
    indices_to_use = np.random.choice(len(x_test), size=1000)
    images_bulk = x_test[indices_to_use]
    expected_digits_bulk = y_test[indices_to_use]

    predictions_bulk = classifier(images=np.array([img.flatten() for img in images_bulk]))

    # Calculate the accuracy of our model
    correct_predictions = np.sum(predictions_bulk == expected_digits_bulk)

    assert (correct_predictions / len(expected_digits_bulk)) > 0.95

def test_classify_digits_interface():
    """
    Test that ClassifyDigits conforms to IClassifyDigits interface.
    
    :param classifier: A ClassifyDigits instance
    """
    # Create a random image array for testing purposes
    images = np.random.rand(1, 28 * 28)

    classifier_instance = ClassifyDigits()
    result = classifier_instance(images=images)
    assert isinstance(result, np.ndarray) and len(result.shape) == 1

def test_classify_digits_model_loading():
    """
    Test that the model is loaded correctly.
    
    :param model_path: Path to our saved Keras model
    """
    try:
        tf.keras.models.load_model(model_path)
    except Exception as e:
        pytest.fail(f"Failed to load model from {model_path}: {str(e)}")
