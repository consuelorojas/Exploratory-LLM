import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL import Image
import tensorflow as tf

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for testing purposes."""
    # For demonstration, we'll use 10 random images with known labels.
    np.random.seed(0)
    images = np.random.rand(10, 28, 28) * 255.0
    labels = np.arange(10)

    return images.astype(np.uint8), labels

@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(classifier, model, test_set):
    """
    Test that the trained digit recognition model achieves more than 95% accuracy.

    Given:
        - A trained digit recognition model.
        - A test set of images with known labels.

    When:
        - The test set is classified using the trained model.

    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Load and prepare the test data
    images, expected_labels = test_set

    # Classify each image in the test set
    predicted_labels = classifier(images)

    # Calculate the accuracy
    correct_predictions = np.sum(predicted_labels == expected_labels)
    accuracy = (correct_predictions / len(expected_labels)) * 100.0

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
