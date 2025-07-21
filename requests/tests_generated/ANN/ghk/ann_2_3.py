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
    # Generate some random 28x28 grayscale images for testing purposes.
    num_images = 10
    images = np.random.randint(0, 256, size=(num_images, 28, 28), dtype=np.uint8)
    
    return images


@pytest.fixture
def classifier():
    """Fixture to create an instance of the ClassifyDigits class."""
    from digits_classifier import ClassifyDigits
    
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, test_set: np.ndarray, classifier: IClassifyDigits):
    """
    Test that the digit recognition model achieves more than 95% accuracy on a sample test set.
    
    Given:
        - A trained digit recognition model
        - A test set of images
    
    When:
        - The test set is classified using the model
    
    Then:
        - An accuracy of more than 95 percent is achieved
    """
    # Generate some random labels for testing purposes (this should be replaced with actual labels).
    num_images = len(test_set)
    labels = np.random.randint(0, 10, size=num_images)

    predictions = classifier(images=test_set)
    
    correct_predictions = sum(predictions == labels)
    accuracy = correct_predictions / num_images
    
    assert accuracy > 0.95
