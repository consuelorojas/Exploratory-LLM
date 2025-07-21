import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
import tensorflow as tf
from PIL import Image


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes."""
    # Replace this with your actual test data loading logic.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (tf.keras.Model): The loaded digit recognition model.
        test_set (tuple[np.ndarray]): A tuple containing images and labels for testing.
        classifier: An instance of IClassifyDigits to classify digits using the model.
    """
    # Extract images and labels from the test set
    images, expected_labels = test_set

    # Normalize and flatten the input data as per the ClassifyDigits implementation
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)

    # Use the classifier to predict digit classes for the given images
    predicted_classes = np.array([int(np.argmax(prediction)) for prediction in model.predict(normalized_images)])

    # Calculate accuracy based on correct predictions and total number of samples
    accuracy = sum(predicted_classes == expected_labels) / len(expected_labels)

    assert (
        accuracy > 0.95
    ), f"Expected the digit recognition model to achieve more than 95% accuracy, but got {accuracy:.2f}"
