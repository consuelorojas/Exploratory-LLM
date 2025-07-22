import sys
sys.path.append('/home/consuelo/Documentos/GitHub/Exploratory-LLM/code')






# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes."""
    # Replace with actual test data loading logic if needed.
    images = np.random.rand(100, 28, 28)
    labels = np.random.randint(0, 10, size=100)
    return images, labels

@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()

def test_digit_recognition_accuracy(model: load_model, test_set, classifier: IClassifyDigits):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (tuple[np.ndarray]): A tuple containing images and labels of the test set.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class for classification.
    """
    # Extract images and labels from the test set
    images, expected_labels = test_set

    # Normalize and flatten images as per the classify method's requirements
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Classify the test set using the model through the classifier instance
    predicted_labels = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy by comparing expected labels with predicted ones
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95.0, f"Expected more than 95% accuracy but got {accuracy:.2f}%"

def test_classifier_accuracy(classifier: IClassifyDigits, test_set):
    """
    Test that the classifier achieves more than 95% accuracy on a given test set.

    Args:
        classifier (IClassifyDigits): An instance of the ClassifyDigits class for classification.
        test_set (tuple[np.ndarray]): A tuple containing images and labels of the test set.
    """
    # Extract images from the test set
    images, expected_labels = test_set

    # Use the classifier to classify the test set directly
    predicted_labels = classifier(images)

    # Calculate accuracy by comparing expected labels with predicted ones
    correct_predictions = sum(1 for pred, exp in zip(predicted_labels, expected_labels) if pred == exp)
    accuracy = (correct_predictions / len(expected_labels)) * 100

    assert accuracy > 95.0, f"Expected more than 95% accuracy but got {accuracy:.2f}%"
