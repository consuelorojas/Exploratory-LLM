import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


@pytest.fixture
def model():
    """Fixture to load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, let's assume we have 10 images with correct labels.
    images = np.random.rand(10, 28 * 28)
    labels = np.array([i % 10 for i in range(10)])
    return images, labels


@pytest.fixture
def classifier():
    """Fixture to create an instance of the ClassifyDigits class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set, classifier):
    """
    Test that the digit recognition model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The trained digit recognition model.
        test_set (tuple): A tuple containing images and their corresponding labels in the test set.
        classifier (ClassifyDigits): An instance of the ClassifyDigits class for classification purposes.
    """
    # Extract images and labels from the test set
    images, labels = test_set

    # Normalize and reshape images as required by the model
    normalized_images = images / 255.0
    reshaped_images = normalized_images.reshape(-1, 28 * 28)

    # Use the classifier to predict digits for each image in the test set
    predictions = np.array([int(np.argmax(prediction)) for prediction in model.predict(reshaped_images)])

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = sum(predictions == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert (
        accuracy > 95.0
    ), f"Expected the digit recognition model to achieve more than 95% accuracy, but got {accuracy:.2f}%"
