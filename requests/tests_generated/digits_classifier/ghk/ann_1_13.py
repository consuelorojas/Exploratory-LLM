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
    """Fixture to load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Fixture to generate a sample test set of images."""
    # For demonstration purposes, we'll use a single image.
    # In a real-world scenario, you'd want to use a larger dataset.
    img = Image.new('L', (28, 28), color=255)
    return np.array([np.array(img)])

@pytest.fixture
def classifier():
    """Fixture to create an instance of the ClassifyDigits class."""
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

    # For demonstration purposes, we'll assume all predictions are correct (i.e., accuracy=100%).
    # In a real-world scenario, you'd want to compare these with actual labels and calculate accuracy accordingly.
    expected_labels = np.array([0])  # Replace with actual labels
    accuracy = np.mean(predictions == expected_labels)

    assert accuracy > 0.95, f"Expected accuracy above 95%, but got {accuracy:.2f}"
