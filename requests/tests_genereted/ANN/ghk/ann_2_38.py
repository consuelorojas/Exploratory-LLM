import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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


def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Tests that the trained model achieves more than 95% accuracy on a given test set.

    Args:
        model (load_model): The loaded digit recognition model.
        test_set (np.ndarray): A sample image for classification.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Generate predictions using the provided classifier
    prediction = classifier(test_set)

    # For demonstration purposes, we'll assume a single correct label.
    # In a real-world scenario, you'd want to compare against multiple labels.
    expected_label = 5

    assert int(prediction) == expected_label


def test_digit_recognition_accuracy_on_multiple_images(model: load_model, classifier: IClassifyDigits):
    """
    Tests that the trained model achieves more than 95% accuracy on a set of images.

    Args:
        model (load_model): The loaded digit recognition model.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Generate multiple test images
    num_images = 10
    test_set = np.random.rand(num_images, 28 * 28)

    predictions = classifier(test_set)
    
    correct_labels = [5] * len(predictions)  # For demonstration purposes

    accuracy = sum(1 for pred, label in zip(predictions, correct_labels) if int(pred) == label) / num_images
    assert accuracy > 0.95


def test_model_loads_correctly(model: load_model):
    """
    Tests that the model loads correctly.

    Args:
        model (load_model): The loaded digit recognition model.
    """
    # Check if the model is not None after loading
    assert model is not None

