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
    # For demonstration purposes only. Replace with actual data.
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

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (tuple): A tuple containing images and labels for testing purposes.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Extract images and labels from the test set
    images, labels = test_set

    # Normalize and flatten the input data
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Make predictions using the classifier instance
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy by comparing actual labels with predicted ones
    correct_predictions = sum(1 for label, pred_label in zip(labels, predicted_labels) if label == pred_label)
    accuracy = (correct_predictions / len(test_set[0])) * 100

    assert (
        accuracy > 95.00
    ), f"Model did not achieve the desired accuracy of more than 95%. Actual accuracy: {accuracy:.2f}%"
