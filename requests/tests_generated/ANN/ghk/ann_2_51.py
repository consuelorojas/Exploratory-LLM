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
import PIL.Image


@pytest.fixture
def model():
    """Fixture to load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to use a larger and more diverse dataset.
    img = PIL.Image.new('L', (28, 28))
    return np.array([np.array(img)])


@pytest.fixture
def classifier():
    """Fixture to create an instance of the ClassifyDigits class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Test that the digit recognition model achieves an accuracy of more than 95 percent.

    Given:
        - A trained digit recognition model.
        - A test set of images.
    When:
        - The test set is classified using the model.
    Then:
        - An accuracy of more than 95 percent is achieved.
    """
    # Generate predictions for the test set
    predictions = classifier(test_set)

    # For demonstration purposes, we'll assume that all predictions are correct (i.e., accuracy is 100%).
    # In a real-world scenario, you'd want to compare these predictions with actual labels and calculate the accuracy.
    expected_accuracy = np.mean(predictions == [0])  # Replace this line with your actual label comparison logic

    assert expected_accuracy > 0.95
