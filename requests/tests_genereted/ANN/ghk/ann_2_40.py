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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)

@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes."""
    # Replace this with your actual test data loading logic.
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
        test_set (tuple[np.ndarray]): A tuple containing images and labels of the test data.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class for classification.
    """
    # Extract images and labels from the test set
    images, labels = test_set

    # Normalize and flatten the input images as per the model's requirements
    normalized_images = images / 255.0
    flattened_images = normalized_images.reshape(-1, 28 * 28)

    # Use the classifier to predict digits for the given images
    predictions = np.array([int(np.argmax(prediction)) for prediction in model.predict(flattened_images)])

    # Calculate accuracy by comparing predicted labels with actual labels
    correct_predictions = sum(predictions == labels)
    accuracy = (correct_predictions / len(labels)) * 100

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
