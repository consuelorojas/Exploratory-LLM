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
from PIL.Image import open, Image
import os


@pytest.fixture
def model():
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set of images for classification."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to generate or load multiple images.
    img_path = os.path.join(os.getcwd(), "tests", "test_image.png")
    return np.array(open(img_path).convert('L').resize((28, 28)))


@pytest.fixture
def classifier():
    """Creates an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set: Image, classifier: IClassifyDigits):
    """
    Tests that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (Image): A sample image for classification.
        classifier (IClassifyDigits): An instance of the digit classification class.
    """

    # Generate predictions using the provided classifier
    prediction = classifier(test_set)

    # For demonstration purposes, we'll assume a single correct label (e.g., 5).
    # In a real-world scenario, you'd want to compare against actual labels for your test set.
    expected_label = np.array([5])

    assert len(prediction) == 1
    accuracy = sum(1 for pred in prediction if pred == expected_label[0]) / len(prediction)

    assert accuracy > 0.95


def test_digit_recognition_output_type(classifier: IClassifyDigits, test_set: Image):
    """
    Tests that the classifier returns an array of integers.

    Args:
        classifier (IClassifyDigits): An instance of the digit classification class.
        test_set (Image): A sample image for classification.
    """

    prediction = classifier(test_set)

    assert isinstance(prediction, np.ndarray)
    assert issubclass(type(prediction[0]), int)


def test_digit_recognition_input_type(classifier: IClassifyDigits):
    """
    Tests that the classifier raises an error when given invalid input.

    Args:
        classifier (IClassifyDigits): An instance of the digit classification class.
    """

    with pytest.raises(TypeError):
        # Attempt to classify a non-numeric value
        classifier("invalid_input")
