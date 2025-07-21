import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent.parent / 'code' / 'ANN'))
from main import ClassifyDigits


# tests/test_digit_recognition.py

import pytest
from digits_classifier.interfaces import IClassifyDigits
from digits_classifier.constants import MODEL_DIGIT_RECOGNITION_PATH
import numpy as np
from PIL.Image import Image
import tensorflow as tf
from sklearn.metrics import accuracy_score


@pytest.fixture
def model():
    """Load the trained digit recognition model."""
    return tf.keras.models.load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set for demonstration purposes.
    
    In a real-world scenario, this would be replaced with an actual test dataset.
    """
    # Generate random images and labels (replace with your own data)
    np.random.seed(0)  # For reproducibility
    num_samples = 100
    image_size = 28 * 28
    images = np.random.rand(num_samples, image_size).astype(np.float32)
    labels = np.random.randint(10, size=num_samples)

    return images / 255.0, labels


@pytest.fixture
def classifier(model):
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(classifier: IClassifyDigits, model, test_set):
    """
    Test that the trained digit recognition model achieves more than 95% accuracy on a given test set.

    Given:
        - A trained digit recognition model
        - A test set

    When:
        - The test set is classified using the model

    Then:
        - An accuracy of more than 95 percent is achieved.
    """
    images, labels = test_set
    predictions = classifier(images)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, predictions)

    assert accuracy > 0.95, f"Expected accuracy to be greater than 0.95 but got {accuracy:.2f}"
