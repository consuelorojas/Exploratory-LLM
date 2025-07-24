import tensorflow as tf
import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration purposes only. Replace with actual test data.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Given:
        - A trained digit recognition model
        - A test set of images and corresponding labels

    When:
        - The test set is classified using the trained model

    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Extract the test data from the fixture
    images, expected_labels = test_set

    # Normalize and flatten the input images for classification
    normalized_images = (images / 255.0).reshape(-1, 28 * 28)

    # Use the classifier to predict labels for the test set
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(normalized_images)])

    # Calculate accuracy based on correct predictions
    accuracy = sum(predicted_labels == expected_labels) / len(expected_labels)

    assert (
        accuracy > 0.95
    ), f"Expected an accuracy of more than 95%, but got {accuracy:.2f}"
