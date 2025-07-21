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
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images for classification."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to generate or load multiple images.
    img_path = os.path.join(os.getcwd(), "tests", "test_image.png")
    return np.array(open(img_path).convert('L').resize((28, 28)))


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Test that the trained model achieves more than 95% accuracy on a given test set.

    Given:
        - A trained digit recognition model.
        - A test set of images for classification.
    When:
        - The test set is classified using the trained model.
    Then:
        - An accuracy of more than 95 percent should be achieved.
    """
    # Generate predictions
    prediction = classifier(test_set)

    # For demonstration purposes, we'll assume a single image with known label (e.g., digit '5').
    expected_label = np.array([5])

    # Calculate accuracy for the test set
    accuracy = np.mean(prediction == expected_label) * 100

    assert accuracy > 95.0


def test_digit_recognition_model_loads(model: load_model):
    """
    Test that the trained model loads successfully.

    Given:
        - A path to a saved digit recognition model.
    When:
        - The model is loaded from file.
    Then:
        - No exceptions should be raised during loading, indicating success.
    """
    assert isinstance(model, tf.keras.Model)


def test_digit_recognition_classifier_initializes(classifier: IClassifyDigits):
    """
    Test that the digit classification class initializes successfully.

    Given:
        - A path to a saved digit recognition model (loaded in classifier).
    When:
        - An instance of the ClassifyDigits class is created.
    Then:
        - No exceptions should be raised during initialization, indicating success.
    """
    assert isinstance(classifier, IClassifyDigits)
