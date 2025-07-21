import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score


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
        classifier: An instance of IClassifyDigits to perform classification.
    """
    # Extract the test data
    images, expected_labels = test_set

    # Normalize and flatten the input images as required by the ClassifyDigits class
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)

    # Perform prediction using the classifier instance directly or through its interface.
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(normalized_images)])

    # Calculate accuracy of predictions against expected labels
    accuracy = accuracy_score(expected_labels, predicted_labels)

    assert (
        accuracy > 0.95
    ), f"Model did not achieve the required accuracy threshold (got {accuracy:.2f}%, needed at least 95%)"
