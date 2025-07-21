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
    # For this example, we'll use random data. In a real scenario,
    # you would load your actual test dataset.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels

@pytest.fixture
def classifier():
    """Create an instance of the ClassifyDigits class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(model, test_set, classifier):
    """
    Test that the digit recognition model achieves more than 95% accuracy.

    Given:
        - A trained digit recognition model.
        - A test set of images and labels.
    When:
        - The test set is classified using the model.
    Then:
        - An accuracy of more than 95 percent is achieved.
    """
    # Load the test data
    images, labels = test_set

    # Normalize and flatten the input data for classification
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Classify the test set using the model
    predictions = classifier(flattened_images)
    
    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = np.sum(predictions == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
