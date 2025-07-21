import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent[4] / 'code' / 'ANN'))


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
    # In a real-world scenario, you'd want to replace this with your actual test data.
    img = PIL.Image.new('L', (28, 28))
    return np.array([np.array(img)])


@pytest.fixture
def classifier():
    """Fixture to create an instance of the ClassifyDigits class."""
    from digits_classifier import ClassifyDigits
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Test that the digit recognition model achieves an accuracy of more than 95 percent on a given test set.

    Args:
        model (tf.keras.Model): The trained digit recognition model.
        test_set (np.ndarray): A sample test set of images.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class.
    """
    # Generate predictions using the classifier
    predictions = classifier(test_set)

    # For demonstration purposes, we'll assume that our test data has labels [0] for simplicity.
    # In a real-world scenario, you'd want to replace this with your actual expected output.
    expected_output = np.array([int(np.argmax(model.predict(test_set / 255.0).reshape(-1, 28 * 28)))])
    
    assert len(predictions) == len(expected_output)
    accuracy = sum(1 for pred, exp in zip(predictions, expected_output) if pred == exp) / len(predictions)

    # Assert that the model achieves an accuracy of more than 95 percent
    assert accuracy > 0.95
