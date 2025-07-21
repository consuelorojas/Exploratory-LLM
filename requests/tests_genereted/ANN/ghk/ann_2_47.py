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

@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generates a sample test set for digit classification."""
    # For demonstration purposes, we'll use a simple test set.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels

@pytest.fixture
def classifier():
    """Creates an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(model: load_model, test_set, classifier: IClassifyDigits):
    """
    Tests that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (tuple): A tuple containing images and labels for testing.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Extract images and labels from the test set
    images, labels = test_set

    # Normalize and flatten the images as required by the model
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Use the classifier to predict digits for the test set
    predictions = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = sum(predictions == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95, f"Expected accuracy to be more than 95%, but got {accuracy:.2f}%"

def test_classifier_accuracy(classifier: IClassifyDigits, test_set):
    """
    Tests that the classifier achieves more than 95% accuracy on a given test set.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
        test_set (tuple): A tuple containing images and labels for testing.
    """
    # Extract images and labels from the test set
    images, labels = test_set

    # Use the classifier to predict digits for the test set
    predictions = classifier(images)

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = sum(predictions == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95, f"Expected accuracy to be more than 95%, but got {accuracy:.2f}%"
