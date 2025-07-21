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
    """Fixture to load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to use a larger dataset.
    img = open("tests/test_image.png").convert('L').resize((28, 28))
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

    # For demonstration purposes, we'll assume the correct labels are [1] (i.e., a single image with label 1).
    # In a real-world scenario, you'd want to use actual labels from your dataset.
    expected_labels = np.array([1])

    # Calculate accuracy
    accuracy = np.sum(predictions == expected_labels) / len(expected_labels)

    assert accuracy > 0.95


def test_model_loads_correctly(model: load_model):
    """
    Test that the model loads correctly.

    Given:
        - A trained digit recognition model.
    When:
        - The model is loaded.
    Then:
        - No errors occur during loading.
    """
    assert isinstance(model, tf.keras.Model)


def test_classifier_initializes_correctly(classifier: IClassifyDigits):
    """
    Test that the classifier initializes correctly.

    Given:
        - A ClassifyDigits instance.
    When:
        - The instance is created.
    Then:
        - No errors occur during initialization.
    """
    assert isinstance(classifier, IClassifyDigits)
