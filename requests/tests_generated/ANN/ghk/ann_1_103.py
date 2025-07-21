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
    """Loads the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generates a sample test set of images with known labels."""
    # For demonstration purposes, we'll use 10 random images.
    num_images = 10
    image_size = (28, 28)
    
    # Generate some dummy data for testing. In real-world scenarios,
    # you would load your actual dataset here.
    test_set_images = np.random.rand(num_images, *image_size).astype(np.uint8)
    test_set_labels = np.random.randint(0, 10, num_images)

    return test_set_images, test_set_labels


@pytest.fixture
def classifier(model):
    """Creates an instance of the digit classification class."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize and flatten the input data.
            normalized = (images / 255.0).reshape(-1, 28 * 28)
            
            predictions = model.predict(normalized)
            return np.array([int(np.argmax(prediction)) for prediction in predictions])

    yield ClassifyDigits()


def test_digit_recognition_accuracy(classifier: IClassifyDigits, test_set):
    """Tests the accuracy of digit recognition."""
    
    # Extract images and labels from the test set.
    test_images, expected_labels = test_set
    
    predicted_labels = classifier(test_images)
    
    correct_predictions = np.sum(predicted_labels == expected_labels)
    total_samples = len(expected_labels)

    assert (correct_predictions / total_samples) > 0.95
