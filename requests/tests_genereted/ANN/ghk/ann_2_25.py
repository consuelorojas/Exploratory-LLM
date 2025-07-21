import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll create 10 random images with labels.
    np.random.seed(0)
    images = np.random.rand(10, 28, 28) * 255.0
    labels = np.random.randint(0, 9, size=10)

    return images.astype(np.uint8), labels

@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(model: load_model, test_set, classifier):
    """
    Test that the trained model achieves more than 95% accuracy on a sample test set.

    Given:
        - A trained digit recognition model.
        - A test set of images with labels.

    When:
        - The test set is classified using the model.

    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Load and prepare data
    images, expected_labels = test_set

    # Classify digits in the test set
    predicted_labels = classifier(images)

    # Calculate accuracy
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100.0

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
