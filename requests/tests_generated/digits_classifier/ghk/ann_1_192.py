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

    # Use the classifier to make predictions on the test set
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = sum(predicted_label == label for predicted_label, label in zip(predicted_labels, labels))
    accuracy = (correct_predictions / len(labels)) * 100

    assert (
        accuracy > 95.0
    ), f"Expected model to achieve more than 95% accuracy but got {accuracy:.2f}%"


def test_digit_recognition_using_classifier(classifier: IClassifyDigits, test_set):
    """
    Test that the classifier achieves more than 95% accuracy on a given test set.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
        test_set (tuple): A tuple containing images and labels for testing purposes.
    """
    # Extract images from the test set
    images, _ = test_set

    # Use the classifier to make predictions on the test set
    predicted_labels = classifier(images)

    # For demonstration purposes only. Replace with actual label data or a more sophisticated method of generating expected output.
    labels = np.random.randint(0, 10, size=len(predicted_labels))

    correct_predictions = sum(predicted_label == label for predicted_label, label in zip(predicted_labels, labels))
    accuracy = (correct_predictions / len(labels)) * 100

    assert (
        accuracy > 95.0
    ), f"Expected classifier to achieve more than 95% accuracy but got {accuracy:.2f}%"
