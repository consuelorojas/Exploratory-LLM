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
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)

    images = np.random.rand(num_samples, *image_size).astype(np.uint8) / 255.0
    labels = np.random.randint(10, size=num_samples)

    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a sample test set.

    Given:
        - A trained digit recognition model
        - A test set of images and their corresponding labels

    When:
        - The test set is classified using the trained model

    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Extract the test data from the fixture
    images, expected_labels = test_set

    # Classify the test set using the classifier instance
    predicted_labels = classifier(images)

    # Calculate the accuracy by comparing the predicted labels with the actual ones
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100.0

    assert (
        accuracy > 95.0
    ), f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
