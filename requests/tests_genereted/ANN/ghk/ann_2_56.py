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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images with known labels."""
    # For simplicity, let's assume we have 10 images in our test set.
    num_images = 10

    # Create an array to store the image data and their corresponding labels.
    images = np.zeros((num_images, 28 * 28))
    labels = np.arange(num_images)

    return images, labels


@pytest.fixture
def classifier():
    """Create a digit classification instance."""
    class ClassifyDigits(IClassifyDigits):
        def __call__(self, images: np.ndarray) -> np.ndarray:
            # Normalize the input data.
            normalized_images = images / 255.0

            # Reshape the input data to match the model's expected shape.
            reshaped_images = normalized_images.reshape(-1, 28 * 28)

            # Use a mock prediction for demonstration purposes only.
            predictions = np.array([np.random.randint(10) for _ in range(len(images))])

            return np.array(predictions)
    return ClassifyDigits()


def test_digit_recognition_accuracy(model: load_model, classifier: IClassifyDigits):
    """Test the accuracy of digit recognition using a trained model."""
    
    # Generate some random images and labels
    num_images = 1000
    
    # Create an array to store the image data and their corresponding labels.
    images = np.random.rand(num_images, 28 * 28)
    labels = np.arange(10).repeat(int(num_images/10))

    predictions = classifier(images)

    accuracy = sum(predictions == labels) / len(labels)

    assert accuracy > 0.95
