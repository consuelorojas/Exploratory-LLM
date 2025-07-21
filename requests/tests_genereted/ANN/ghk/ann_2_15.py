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
    """Load the trained digit recognition model."""
    return load_model(MODEL_DIGIT_RECOGNITION_PATH)


@pytest.fixture
def test_set():
    """Generate a sample test set of images and their corresponding labels."""
    # For demonstration purposes, we'll use random data.
    np.random.seed(0)
    num_samples = 100
    image_size = (28, 28)

    images = np.random.rand(num_samples, *image_size).astype(np.uint8)
    labels = np.random.randint(10, size=num_samples)

    return images, labels


@pytest.fixture
def classifier():
    """Create an instance of the digit classification class."""
    from digits_classifier import ClassifyDigits

    return ClassifyDigits()


def test_digit_recognition_accuracy(model: tf.keras.Model, test_set, classifier: IClassifyDigits):
    """
    Test that the trained model achieves more than 95% accuracy on a sample test set.

    Args:
        model (tf.keras.Model): The loaded digit recognition model.
        test_set (tuple[np.ndarray, np.ndarray]): A tuple containing images and labels of the test set.
        classifier (IClassifyDigits): An instance of the ClassifyDigits class for classification.
    """
    # Extract images and labels from the test set
    images, expected_labels = test_set

    # Convert PIL Image to numpy array if necessary
    converted_images = np.array([PIL.Image.fromarray(image).convert('L').resize((28, 28)) for image in images])

    # Classify the test set using the model and classifier instance
    predicted_labels = classifier(converted_images)

    # Calculate accuracy by comparing expected labels with predicted ones
    correct_predictions = np.sum(expected_labels == predicted_labels)
    accuracy = (correct_predictions / len(test_set[0])) * 100

    assert accuracy > 95, f"Expected an accuracy of more than 95%, but got {accuracy:.2f}%"
